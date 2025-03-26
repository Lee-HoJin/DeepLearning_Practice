import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import torch
torch.cuda.init()
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
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

# train_dataset = SpamDataset(X_train_tensor, y_train_tensor)
test_dataset = SpamDataset(X_test_tensor, y_test_tensor)

# train/validation 분할 (80:20)
dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)
test_loader = DataLoader(test_dataset, batch_size=64)

# 모델 정의
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, kernel_size, dropout_ratio):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_ratio)
        # nn.Conv1d의 in_channels는 임베딩 차원입니다.
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=kernel_size)
        # GlobalMaxPooling1D는 torch.max()를 사용해 시퀀스 차원에 대해 풀링합니다.
        self.fc = nn.Linear(num_filters, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch_size, sequence_length]
        x = self.embedding(x)  # -> [batch_size, sequence_length, embedding_dim]
        x = self.dropout(x)
        # Conv1d는 [batch, channels, seq_len] 형태의 텐서를 기대하므로 차원 변환 필요
        x = x.permute(0, 2, 1)  # -> [batch_size, embedding_dim, sequence_length]
        x = F.relu(self.conv1d(x))  # -> [batch_size, num_filters, L_out]
        # Global max pooling: 시퀀스 길이(L_out) 차원에 대해 최대값 선택
        x = torch.max(x, dim=2)[0]  # -> [batch_size, num_filters]
        x = self.dropout(x)
        x = self.fc(x)  # -> [batch_size, 1]
        x = self.sigmoid(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier(vocab_size = tokenizer.vocab_size,
                      embedding_dim=32,
                      num_filters = 32,
                      kernel_size=5,
                      dropout_ratio=0.3).to(device)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

patience = 4
best_val_acc = 0
trigger_times = 0

# 학습
train_losses = []
val_losses = []
for epoch in range(10):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    epoch_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs).view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        predicted = (outputs >= 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_acc = correct / total
    train_losses.append(epoch_loss / len(train_loader))
    print(f"학습 정확도 epoch {epoch + 1}: {(correct / total)}")
    
    # model.eval()
    # val_loss = 0
    # correct_val = 0
    # total_val = 0
    # with torch.no_grad():
    #     for inputs, labels in val_loader:
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         outputs = model(inputs)
    #         loss = criterion(outputs, labels)
    #         val_loss += loss.item() * inputs.size(0)
    #         predicted = (outputs >= 0.5).float()
    #         total_val += labels.size(0)
    #         correct_val += (predicted == labels).sum().item()
    # val_acc = correct_val / total_val
    # val_loss = val_loss / total_val
    # val_losses.append(val_loss / len(val_loader))
    
    # if val_acc > best_val_acc:
    #     best_val_acc = val_acc
    #     trigger_times = 0
    #     torch.save(model.state_dict(), "11-4-2_best_model.pt")
    #     print("모델 저장!")
    # else:
    #     trigger_times += 1
    #     if trigger_times >= patience:
    #         print("Early Stopping!")
    #         break

# 테스트 정확도
# model.load_state_dict(torch.load("11-4-2_best_model.pt"))
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs).view(-1)
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
plt.savefig('11-4-2_spam_classify_torch.png')