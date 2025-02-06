from pykrx import stock
import ta
import ta.momentum

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
torch.cuda.init()
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import pandas as pd
import os

torch.manual_seed(777)  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')

# Train Parameters
num_layers = 4  # number of layers in RNN
learning_rate = 0.0005
num_epochs = 5000
input_size = 8
hidden_size = 8
num_classes = 1
timesteps = seq_length = 60
future_seq = 15  # 예측하고자 하는 미래 시퀀스 길이

weight_decay = 1e-4
early_stopping_patience = 500
early_stopping_delta = 1e-4

start_date = "20150101"
end_date = "20250204"
code = "035420"
stock_name = 'SK하이닉스'

## 새 종목을 훈련할 때만 True
FLAG = False

if FLAG:
    df = stock.get_market_ohlcv_by_date(start_date, end_date, code)
    print(df.head())

    # adding RSI Index
    df['RSI'] = ta.momentum.rsi(df['종가'], window = 14)

    df_trading = stock.get_market_trading_value_by_date(start_date, end_date, code)

    df_close = df['종가']
    df = df.drop(columns=['종가'])

    df_last_actual_price = df_close[-future_seq:]
    # print(df_last_actual_price.to_numpy())

    df_combined = pd.concat([df, df_trading, df_close], axis = 1)

    df_combined.to_csv(f"./{stock_name}.csv", encoding="utf-8-sig")  # utf-8-sig는 한글 깨짐 방지용

df = pd.read_csv(f'{stock_name}.csv', encoding='utf-8-sig')

df_last_actual_price = df['종가'][-future_seq:]


# 필요 없는 행 제거
df = df.drop(columns=['날짜', '등락률', '기타법인', '개인', '전체'])

# 마지막 실제 데이터 제외
df = df.iloc[:-future_seq]
df = df.apply(pd.to_numeric, errors='coerce')
df = df.dropna()
xy = df.to_numpy()

# 종가 열의 최소/최대 값 저장 (inverse scaling용)
target_min = np.min(xy[:, -1])
target_max = np.max(xy[:, -1])

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents zero division
    return numerator / (denominator + 1e-7)

xy = MinMaxScaler(xy)
x = xy
y = xy[:, [-1]]  # 종가 열 (shape: [samples, 1])

# Build dataset using sliding window (many-to-many)
# 각 sample의 입력은 seq_length일, 타겟은 그 다음 future_seq일치 데이터
dataX = []
dataY = []
for i in range(0, len(y) - seq_length - future_seq + 1):
    _x = x[i:i + seq_length]           # shape: [seq_length, input_size]
    _y = y[i + seq_length : i + seq_length + future_seq]  # shape: [future_seq, 1]
    dataX.append(_x)
    dataY.append(_y)

# Device Selection
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"\nUsing {device} device")
print("GPU: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# Train/Test split
train_size = int(len(dataX) * 0.8)
test_size = len(dataX) - train_size

trainX = torch.Tensor(np.array(dataX[0:train_size])).to(device)  # shape: [train_samples, seq_length, input_size]
testX = torch.Tensor(np.array(dataX[train_size:])).to(device)
trainY = torch.Tensor(np.array(dataY[0:train_size])).to(device)  # shape: [train_samples, future_seq, 1]
testY = torch.Tensor(np.array(dataY[train_size:])).to(device)

# --------------------------
# 1. EarlyStopping 클래스 정의
# --------------------------
class EarlyStopping:
    def __init__(self, patience=200, delta=0.0001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# -------------------------------
# 2. LSTM 모델 (Many-to-Many 추가)
# -------------------------------
class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, future_seq):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.future_seq = future_seq

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        # 최종 FC layer: 출력 차원 = num_classes * future_seq
        self.fc = nn.Linear(hidden_size, num_classes * future_seq)

    def forward(self, x):
        # Initialize hidden and cell states
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)

        # LSTM 순전파
        _, (h_out, _) = self.lstm(x, (h_0, c_0))
        # h_out shape: [num_layers, batch_size, hidden_size] -> 마지막 레이어 사용
        h_out = h_out[-1]  # shape: [batch_size, hidden_size]
        out = self.fc(h_out)  # shape: [batch_size, num_classes * future_seq]
        # reshape해서 many-to-many 형태로 변환 (예: [batch_size, future_seq, num_classes])
        out = out.view(-1, self.future_seq, self.num_classes)
        return out

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Instantiate and initialize model
lstm = LSTM(num_classes, input_size, hidden_size, num_layers, future_seq)
lstm.apply(init_weights)
lstm = lstm.to(device)

if not FLAG:
    lstm.load_state_dict(torch.load('model.pth'))
    print("model loaded")

# Loss and optimizer
criterion = torch.nn.MSELoss()  # Mean Squared Error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Setup EarlyStopping
early_stopping = EarlyStopping(patience=early_stopping_patience, delta=early_stopping_delta)

# Training Loop with EarlyStopping
for epoch in range(num_epochs):
    lstm.train()
    optimizer.zero_grad()
    outputs = lstm(trainX)  # outputs shape: [batch, future_seq, num_classes]
    loss = criterion(outputs, trainY)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(lstm.parameters(), max_norm=1.0)
    optimizer.step()

    # Compute validation loss on test set
    lstm.eval()
    with torch.no_grad():
        val_outputs = lstm(testX)
        val_loss = criterion(val_outputs, testY)
    lstm.train()  # switch back to training mode

    early_stopping(val_loss.item())

    if epoch % 200 == 0:
        print(f"Epoch: {epoch}, training loss: {loss.item():.8f}, validation loss: {val_loss.item():.8f}")
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch}")
        break

print("Learning finished!")

# Testing the model
lstm.eval()
test_predict = lstm(testX)  # shape: [test_samples, future_seq, num_classes]

# Next future prediction using the latest time window
# 최근 seq_length일 데이터를 입력으로 하여 앞으로 future_seq일 치 예측
last_window_index = seq_length + future_seq
last_window = xy[-last_window_index:]  # shape: [seq_length, input_size]
print(last_window[-1:])
print("MAX: ", target_max)
print("MIN: ", target_min)
last_window_tensor = torch.Tensor(last_window).unsqueeze(0).to(device)  # add batch dimension

lstm.eval()
with torch.no_grad():
    next_pred = lstm(last_window_tensor)  # shape: [1, future_seq, num_classes]

# Inverse scaling for prediction (단, num_classes=1로 가정)
next_pred_original = next_pred * (target_max - target_min) + target_min
print("Next future predicted prices (original scale):\n", next_pred_original)

last_prediction = next_pred_original.to('cpu').view(future_seq, num_classes)

plt.title(f"{stock_name}")
plt.plot(df_last_actual_price.to_numpy(), label="actual price")
plt.plot(last_prediction, label="predicted price")
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.legend()
plt.savefig("./stock_prediction_torch_many_to_many.png")

if not FLAG:
    torch.save(lstm.state_dict(), './model.pth')