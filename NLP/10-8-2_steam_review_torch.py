import pandas as pd
import numpy as np
import urllib.request
import re
from collections import Counter
from konlpy.tag import Mecab
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

# 1. 데이터 다운로드 및 로드
urllib.request.urlretrieve("https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/steam.txt", filename="steam.txt")
total_data = pd.read_table('steam.txt', names=['label', 'reviews'])
total_data.drop_duplicates(subset=['reviews'], inplace=True)  # 중복 제거

# 2. train / test 분할 및 전처리
train_data, test_data = train_test_split(total_data, test_size=0.25, random_state=42)

# 한글과 공백 이외 제거, 빈 문자열 제거
def clean_text(df, column):
    df[column] = df[column].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "", regex=True)
    df[column].replace('', np.nan, inplace=True)
    df.dropna(subset=[column], inplace=True)
    return df

train_data = clean_text(train_data, 'reviews')
test_data = clean_text(test_data, 'reviews')
test_data.drop_duplicates(subset=['reviews'], inplace=True)

# 3. 토큰화 및 불용어 제거 (Mecab 사용)
stopwords = ['도', '는', '다', '의', '가', '이', '은', '한', '에', '하', '고', 
             '을', '를', '인', '듯', '과', '와', '네', '들', '듯', '지', '임', '게', '만', '게임', '겜', '되', '음', '면']
mecab = Mecab()

train_data['tokenized'] = train_data['reviews'].apply(lambda x: [word for word in mecab.morphs(x) if word not in stopwords])
test_data['tokenized'] = test_data['reviews'].apply(lambda x: [word for word in mecab.morphs(x) if word not in stopwords])

X_train_raw = train_data['tokenized'].values
y_train = train_data['label'].values
X_test_raw = test_data['tokenized'].values
y_test = test_data['label'].values

# 4. 단어 집합 구축 (threshold = 2)
def build_vocab(tokenized_texts, threshold=2):
    counter = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)
    total_cnt = len(counter)
    rare_cnt = sum(1 for token, freq in counter.items() if freq < threshold)
    rare_freq = sum(freq for token, freq in counter.items() if freq < threshold)
    print('단어 집합(vocabulary)의 크기 :', total_cnt)
    print('등장 빈도가 {}번 이하인 희귀 단어의 수: {}'.format(threshold - 1, rare_cnt))
    print("단어 집합에서 희귀 단어의 비율: {:.2f}".format((rare_cnt / total_cnt)*100))
    print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율: {:.2f}".format((rare_freq / sum(counter.values()))*100))
    
    # PAD:0, OOV:1 로 예약. 빈도수 threshold 이상인 단어만 포함.
    word2index = {'PAD': 0, 'OOV': 1}
    idx = 2
    for word, freq in counter.items():
        if freq >= threshold:
            word2index[word] = idx
            idx += 1
    vocab_size = idx  # 전체 단어 개수 (인덱스 개수)
    print('최종 단어 집합의 크기 :', vocab_size)
    return word2index

threshold = 2
word2index = build_vocab(X_train_raw, threshold)
vocab_size = len(word2index)  # Embedding layer에 넣을 단어 수

# 5. 토큰 시퀀스를 인덱스 시퀀스로 변환하는 함수
def texts_to_sequences(tokenized_texts, word2index):
    sequences = []
    for tokens in tokenized_texts:
        seq = [word2index.get(token, word2index['OOV']) for token in tokens]
        sequences.append(seq)
    return sequences

X_train_seq = texts_to_sequences(X_train_raw, word2index)
X_test_seq = texts_to_sequences(X_test_raw, word2index)

# 6. 패딩 (최대 길이 max_len = 60)
max_len = 60
def pad_sequences_custom(sequences, max_len, padding='post'):
    padded = []
    for seq in sequences:
        if len(seq) < max_len:
            if padding=='post':
                seq = seq + [0]*(max_len - len(seq))
            else:
                seq = [0]*(max_len - len(seq)) + seq
        else:
            seq = seq[:max_len]
        padded.append(seq)
    return np.array(padded)

X_train = pad_sequences_custom(X_train_seq, max_len)
X_test = pad_sequences_custom(X_test_seq, max_len)

# 7. PyTorch 데이터셋 및 DataLoader 구성
X_train_tensor = torch.LongTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)  # (N, 1)
X_test_tensor = torch.LongTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).unsqueeze(1)

# train/validation 분할 (80:20)
dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

batch_size = 256
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# 8. PyTorch 모델 구성 (Embedding → Bidirectional LSTM → Dense)
embedding_dim = 100
hidden_units = 128

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_units):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_units, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_units*2, 1)
        
    def forward(self, x):
        x = self.embedding(x)  # (batch, seq_len, embedding_dim)
        output, (h_n, c_n) = self.lstm(x)
        # h_n: (num_directions, batch, hidden_units) → concatenate 양방향 마지막 hidden state
        h_n_cat = torch.cat((h_n[0], h_n[1]), dim=1)  # (batch, hidden_units*2)
        out = self.fc(h_n_cat)
        out = torch.sigmoid(out)
        return out

model = SentimentModel(vocab_size, embedding_dim, hidden_units)

# 9. 학습 설정 및 학습 루프 (Early Stopping & 모델 체크포인트)
criterion = nn.BCELoss()
optimizer = optim.RMSprop(model.parameters())

num_epochs = 15
patience = 4
best_val_acc = 0
trigger_times = 0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        predicted = (outputs >= 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_acc = correct / total
    train_loss = train_loss / total
    
    model.eval()
    val_loss = 0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            predicted = (outputs >= 0.5).float()
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    val_acc = correct_val / total_val
    val_loss = val_loss / total_val
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, " +
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # 모델 성능이 개선되면 저장, 개선 없으면 patience 카운트 증가
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        trigger_times = 0
        torch.save(model.state_dict(), "best_model.pt")
        print("모델 저장!")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early Stopping!")
            break

# 10. 저장된 모델 로드 후 테스트 평가
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
correct_test = 0
total_test = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = (outputs >= 0.5).float()
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()
test_acc = correct_test / total_test
print("테스트 정확도: {:.4f}".format(test_acc))

# 11. 새로운 문장에 대한 감성 예측 함수
def sentiment_predict(new_sentence):
    new_sentence = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣 ]','', new_sentence)
    tokens = mecab.morphs(new_sentence)
    tokens = [token for token in tokens if token not in stopwords]
    # 토큰을 인덱스 시퀀스로 변환 (OOV 처리)
    seq = [word2index.get(token, word2index['OOV']) for token in tokens]
    # 패딩
    if len(seq) < max_len:
        seq = seq + [word2index['PAD']] * (max_len - len(seq))
    else:
        seq = seq[:max_len]
    seq_tensor = torch.LongTensor([seq])
    model.eval()
    with torch.no_grad():
        output = model(seq_tensor)
        score = output.item()
    if score > 0.5:
        print("{:.2f}% 확률로 긍정 리뷰입니다.".format(score * 100))
    else:
        print("{:.2f}% 확률로 부정 리뷰입니다.".format((1 - score) * 100))

# 예측 예시
sentiment_predict('노잼 ..완전 재미 없음 ㅉㅉ')
sentiment_predict('조금 어렵지만 재밌음ㅋㅋ')
