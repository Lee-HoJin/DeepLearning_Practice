import os
import re
import unicodedata
import numpy as np
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 하이퍼파라미터
num_samples = 33000
embedding_dim = 64
hidden_dim = 64
batch_size = 128
n_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 전처리 함수 ---
def to_ascii(s):
    # 프랑스어 악센트(accent) 삭제 (예: 'déjà diné' -> 'deja dine')
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def preprocess_sentence(sent):
    sent = to_ascii(sent.lower())
    # 단어와 구두점 사이에 공백 추가 ("I am a student." -> "I am a student .")
    sent = re.sub(r"([?.!,¿])", r" \1", sent)
    # (a-z, A-Z, ".", "?", "!", ",") 외 문자는 공백으로 변환
    sent = re.sub(r"[^a-zA-Z!.?]+", r" ", sent)
    # 다수 공백을 하나로 치환
    sent = re.sub(r"\s+", " ", sent).strip()
    return sent

# 전처리 테스트
en_sent = "Have you had dinner?"
fr_sent = "Avez-vous déjà diné?"
print('전처리 전 영어 문장 :', en_sent)
print('전처리 후 영어 문장 :', preprocess_sentence(en_sent))
print('전처리 전 프랑스어 문장 :', fr_sent)
print('전처리 후 프랑스어 문장 :', preprocess_sentence(fr_sent))

# --- 데이터 로드 ---
def load_preprocessed_data():
    encoder_input, decoder_input, decoder_target = [], [], []
    with open("./fra-eng/fra.txt", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            # source, target, 나머지 (무시)
            src_line, tar_line, _ = line.strip().split('\t')
            # 전처리: source는 단어 리스트
            src_line = preprocess_sentence(src_line).split()
            # target: <sos>와 <eos> 추가
            tar_line_clean = preprocess_sentence(tar_line)
            tar_line_in = ("<sos> " + tar_line_clean).split()
            tar_line_out = (tar_line_clean + " <eos>").split()
            encoder_input.append(src_line)
            decoder_input.append(tar_line_in)
            decoder_target.append(tar_line_out)
            if i == num_samples - 1:
                break
    return encoder_input, decoder_input, decoder_target

sents_en_in, sents_fra_in, sents_fra_out = load_preprocessed_data()
print('인코더의 입력 샘플 :', sents_en_in[:5])
print('디코더의 입력 샘플 :', sents_fra_in[:5])
print('디코더의 레이블 샘플 :', sents_fra_out[:5])

# --- 단어 집합 생성 ---
def build_vocab(sentences):
    counter = Counter()
    for sent in sentences:
        counter.update(sent)
    # 인덱스 0은 padding으로 예약
    word2index = {word: idx+1 for idx, word in enumerate(sorted(counter))}
    index2word = {idx: word for word, idx in word2index.items()}
    return word2index, index2word

# 영어: 인코더 데이터에서 단어 집합 생성
src_to_index, index_to_src = build_vocab(sents_en_in)
src_vocab_size = len(src_to_index) + 1  # padding 0 포함

# 프랑스어: 디코더 입력과 레이블 모두 사용 (두 데이터셋을 합쳐 학습)
all_fra = sents_fra_in + sents_fra_out
tar_to_index, index_to_tar = build_vocab(all_fra)
tar_vocab_size = len(tar_to_index) + 1

print("영어 단어 집합의 크기 : {:d}, 프랑스어 단어 집합의 크기 : {:d}".format(src_vocab_size, tar_vocab_size))

# --- 시퀀스를 정수 시퀀스로 변환 ---
def texts_to_sequences(sentences, word2index):
    return [[word2index[word] for word in sent] for sent in sentences]

encoder_seq = texts_to_sequences(sents_en_in, src_to_index)
decoder_input_seq = texts_to_sequences(sents_fra_in, tar_to_index)
decoder_target_seq = texts_to_sequences(sents_fra_out, tar_to_index)

# --- 시퀀스 패딩 ---
def pad_sequences_custom(sequences, padding="post"):
    max_len = max(len(seq) for seq in sequences)
    padded = []
    for seq in sequences:
        if padding == "post":
            padded_seq = seq + [0]*(max_len - len(seq))
        else:
            padded_seq = [0]*(max_len - len(seq)) + seq
        padded.append(padded_seq)
    return np.array(padded)

encoder_input = pad_sequences_custom(encoder_seq, padding="post")
decoder_input = pad_sequences_custom(decoder_input_seq, padding="post")
decoder_target = pad_sequences_custom(decoder_target_seq, padding="post")

print('인코더 입력의 크기:', encoder_input.shape)
print('디코더 입력의 크기:', decoder_input.shape)
print('디코더 레이블의 크기:', decoder_target.shape)

# --- 데이터 셔플 및 분할 ---
indices = np.arange(encoder_input.shape[0])
np.random.shuffle(indices)

encoder_input = encoder_input[indices]
decoder_input = decoder_input[indices]
decoder_target = decoder_target[indices]

n_val = int(num_samples * 0.1)
encoder_input_train = encoder_input[:-n_val]
decoder_input_train = decoder_input[:-n_val]
decoder_target_train = decoder_target[:-n_val]

encoder_input_test = encoder_input[-n_val:]
decoder_input_test = decoder_input[-n_val:]
decoder_target_test = decoder_target[-n_val:]

print('훈련 source 데이터:', encoder_input_train.shape)
print('훈련 target 입력 데이터:', decoder_input_train.shape)
print('훈련 target 레이블 데이터:', decoder_target_train.shape)
print('테스트 source 데이터:', encoder_input_test.shape)
print('테스트 target 입력 데이터:', decoder_input_test.shape)
print('테스트 target 레이블 데이터:', decoder_target_test.shape)

# --- PyTorch Dataset ---
class TranslationDataset(Dataset):
    def __init__(self, encoder_inputs, decoder_inputs, decoder_targets):
        self.encoder_inputs = torch.tensor(encoder_inputs, dtype=torch.long)
        self.decoder_inputs = torch.tensor(decoder_inputs, dtype=torch.long)
        self.decoder_targets = torch.tensor(decoder_targets, dtype=torch.long)
    def __len__(self):
        return len(self.encoder_inputs)
    def __getitem__(self, idx):
        return (self.encoder_inputs[idx],
                self.decoder_inputs[idx],
                self.decoder_targets[idx])

train_dataset = TranslationDataset(encoder_input_train, decoder_input_train, decoder_target_train)
test_dataset  = TranslationDataset(encoder_input_test, decoder_input_test, decoder_target_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size)

# --- 모델 정의 ---
# 인코더
class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
    def forward(self, x):
        # x: [batch, seq_len]
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(embedded)
        return hidden, cell

# 디코더
class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    def forward(self, x, hidden, cell):
        # x: [batch] -> [batch, 1]
        x = x.unsqueeze(1)
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc(output.squeeze(1))  # [batch, output_dim]
        return prediction, hidden, cell

# Seq2Seq (Teacher Forcing 방식)
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # src: [batch, src_len], trg: [batch, trg_len]
        batch_size = src.size(0)
        trg_len = trg.size(1)
        trg_vocab_size = self.decoder.embedding.num_embeddings

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        hidden, cell = self.encoder(src)
        # 첫 입력: <sos> 토큰 (trg의 첫 토큰)
        input_token = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell)
            outputs[:, t] = output
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t] if teacher_force else top1
        return outputs

# 인스턴스 생성
encoder = Encoder(src_vocab_size, embedding_dim, hidden_dim)
decoder = Decoder(tar_vocab_size, embedding_dim, hidden_dim)
model = Seq2Seq(encoder, decoder, device).to(device)

# 손실함수와 옵티마이저 (padding index 0은 무시)
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 학습 루프 ---
model.train()
for epoch in range(1, n_epochs+1):
    epoch_loss = 0
    for src_batch, trg_batch, trg_gold in train_loader:
        src_batch = src_batch.to(device)
        trg_batch = trg_batch.to(device)
        trg_gold = trg_gold.to(device)

        optimizer.zero_grad()
        output = model(src_batch, trg_batch, teacher_forcing_ratio=0.5)
        # output: [batch, trg_len, tar_vocab_size]
        # loss 계산 시, t=0은 건너뜁니다 (<sos>에 해당)
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg_gold = trg_gold[:, 1:].reshape(-1)
        loss = criterion(output, trg_gold)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch} Loss: {epoch_loss/len(train_loader):.4f}")

# --- 추론(디코딩) 함수 ---
def decode_sequence(model, input_seq):
    model.eval()
    with torch.no_grad():
        src_tensor = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
        hidden, cell = model.encoder(src_tensor)
        input_token = torch.tensor([tar_to_index['<sos>']], dtype=torch.long).to(device)
        decoded_tokens = []
        for t in range(50):
            output, hidden, cell = model.decoder(input_token, hidden, cell)
            top1 = output.argmax(1).item()
            token = index_to_tar[top1]
            if token == '<eos>':
                break
            decoded_tokens.append(token)
            input_token = torch.tensor([top1], dtype=torch.long).to(device)
    return " ".join(decoded_tokens)

# --- 시퀀스를 단어로 변환하는 함수 ---
def seq_to_src(input_seq):
    sentence = []
    for idx in input_seq:
        if idx != 0:
            sentence.append(index_to_src[idx])
    return " ".join(sentence)

def seq_to_tar(input_seq):
    sentence = []
    for idx in input_seq:
        # <sos>와 <eos>는 제외
        if idx != 0 and idx != tar_to_index.get('<sos>', -1) and idx != tar_to_index.get('<eos>', -1):
            sentence.append(index_to_tar[idx])
    return " ".join(sentence)

# --- 추론 예시 ---
# 훈련 데이터에서 몇 개 추출하여 번역 결과 확인
print("\n[훈련 데이터 추론]")
for seq_index in [3, 50, 100, 300, 1001]:
    input_seq = encoder_input_train[seq_index]
    input_seq_batch = input_seq[np.newaxis, :]  # shape: [1, seq_len]
    decoded_sentence = decode_sequence(model, input_seq)
    print("입력문장 :", seq_to_src(input_seq))
    print("정답문장 :", seq_to_tar(decoder_input_train[seq_index]))
    print("번역문장 :", decoded_sentence)
    print("-"*50)

print("\n[테스트 데이터 추론]")
for seq_index in [3, 50, 100, 300, 1001]:
    input_seq = encoder_input_test[seq_index]
    input_seq_batch = input_seq[np.newaxis, :]
    decoded_sentence = decode_sequence(model, input_seq)
    print("입력문장 :", seq_to_src(input_seq))
    print("정답문장 :", seq_to_tar(decoder_input_test[seq_index]))
    print("번역문장 :", decoded_sentence)
    print("-"*50)
