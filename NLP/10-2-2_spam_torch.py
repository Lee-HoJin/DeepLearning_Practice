import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import torch
torch.cuda.init()
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# 데이터 로딩 및 전처리
data = pd.read_csv('spam.csv', encoding='latin1')
data = data[['v1', 'v2']]
data['v1'] = data['v1'].map({'ham': 0, 'spam': 1})
data.drop_duplicates(subset=['v2'], inplace=True)

X_data = data['v2']
y_data = data['v1']

X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, stratify=y_data)

# Hugging Face tokenizer 사용
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

# 리스트로 변환
X_train = list(X_train)
X_test = list(X_test)

# 먼저 토큰화 (tokenizer.tokenize)해서 각 문장의 길이 측정
token_lens = [len(tokenizer.tokenize(text)) for text in X_train]
max_length = max(token_lens)

# print("훈련 데이터에서 가장 긴 문장의 토큰 수:", max_length)

# 토큰화 및 인코딩
train_tokens = tokenizer(
    X_train,
    padding='longest',
    truncation=True,
    return_tensors='pt'
    )
test_tokens = tokenizer(
    X_test,
    padding='longest',
    truncation=True,
    return_tensors='pt'
    )

# Tensor 준비
X_train_tensor = train_tokens['input_ids']
X_test_tensor = test_tokens['input_ids']
y_train_tensor = torch.FloatTensor(y_train.values)
y_test_tensor = torch.FloatTensor(y_test.values)

# Dataset 정의
class SpamDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = SpamDataset(X_train_tensor, y_train_tensor)
test_dataset = SpamDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# 모델 정의
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        # 실제 토큰 길이에 맞게 시퀀스를 패킹합니다.
        packed_embedded = rnn_utils.pack_padded_sequence(
            embedded, 
            lengths.cpu(), 
            batch_first=True, 
            enforce_sorted=False
        )
        packed_output, hidden = self.rnn(packed_embedded)
        # hidden은 [num_layers, batch, hidden_size]의 형태이므로 마지막 layer의 hidden state 사용
        out = self.fc(hidden[-1])
        return self.sigmoid(out)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RNNClassifier(vocab_size = tokenizer.vocab_size, embedding_dim=64, hidden_size=64).to(device)

criterion = nn.BCELoss()
optimizer = optim.RMSprop(model.parameters(), lr=0.001)

# 학습
train_losses = []
for epoch in range(10):
    model.train()
    epoch_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        lengths = (inputs != tokenizer.pad_token_id).sum(dim=1)

        optimizer.zero_grad()
        outputs = model(inputs, lengths).view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    train_losses.append(epoch_loss / len(train_loader))

# 테스트 정확도
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        lengths = (inputs != tokenizer.pad_token_id).sum(dim=1)


        outputs = model(inputs, lengths).view(-1)
        predicted = (outputs >= 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print("테스트 정확도: %.8f" % (correct / total))

# 손실 시각화
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('10-2_spam_classify_torch.png')