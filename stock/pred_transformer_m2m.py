import sys
import math
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
    
def get_stock_code_by_name(name: str) -> str:
    """종목 이름을 입력하면 종목 코드를 반환"""
    stock_list = stock.get_market_ticker_list(market="ALL")  # 코스피 & 코스닥 전체
    for code in stock_list:
        if stock.get_market_ticker_name(code) == name:
            return code
    return None  # 해당하는 종목이 없을 경우 None 반환

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("종목명을 함께 입력해주세요\npython3 pred_t.py [종목명]")
        sys.exit(1)

    stock_name = sys.argv[1]  # 터미널에서 입력한 종목명
    code = get_stock_code_by_name(stock_name)

    if code:
        print(f"종목명: {stock_name}, 종목 코드: {code}")
    else:
        print(f"'{stock_name}'에 해당하는 종목 코드를 찾을 수 없습니다.")


# Train Parameters
num_layers = 2  # number of layers in RNN
learning_rate = 0.0005
num_epochs = 5000
input_size = 8
num_classes = 1
timesteps = seq_length = 30
future_seq = 15  # 예측하고자 하는 미래 시퀀스 길이

d_model = 32         # 내부 임베딩 차원
nhead = 8            # 멀티헤드 Attention의 head 수
dropout = 0.1

weight_decay = 1e-5
early_stopping_patience = 500
early_stopping_delta = 1e-4

start_date = "20150101"
end_date = "20250204"

file_path = f"./stocks/{stock_name}.csv"

if not os.path.exists(file_path):
    print(f"파일이 존재하지 않습니다. {stock_name} 데이터를 다운로드합니다...")

    df = stock.get_market_ohlcv_by_date(start_date, end_date, code)
    # print(df.head())

    # adding RSI Index
    df['RSI'] = ta.momentum.rsi(df['종가'], window = 14)

    df_trading = stock.get_market_trading_value_by_date(start_date, end_date, code)

    df_close = df['종가']
    df = df.drop(columns=['종가'])

    df_combined = pd.concat([df, df_trading, df_close], axis = 1)

    df_combined.to_csv(f"./stocks/{stock_name}.csv", encoding="utf-8-sig")  # utf-8-sig는 한글 깨짐 방지용
    print(f"{stock_name} 데이터 저장 완료!")

df = pd.read_csv(file_path, encoding='utf-8-sig')
print(df.head(3))

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

# Positional Encoding 모듈 (공식 예제 참고)
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """
        d_model: 임베딩 차원
        dropout: 드롭아웃 확률
        max_len: 최대 시퀀스 길이
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 위치 인코딩 행렬 초기화
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 주기적 함수를 위한 분모 계수
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # 짝수 인덱스에 sin 적용
        pe[:, 1::2] = torch.cos(position * div_term)  # 홀수 인덱스에 cos 적용
        pe = pe.unsqueeze(0)  # 배치 차원 추가
        self.register_buffer('pe', pe)  # 학습 파라미터로 등록하지 않음

    def forward(self, x):
        """
        x: [batch_size, seq_length, d_model]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# Transformer 기반 시계열 예측 모델
class TransformerModel(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, num_classes, future_seq, dropout=0.1):
        """
        input_size: 입력 피처 수 (예: 시계열의 각 시점에 대한 피처 수)
        d_model: Transformer 내부 임베딩 차원
        nhead: 멀티헤드 Attention의 head 수
        num_layers: Transformer Encoder의 레이어 수
        num_classes: 출력 차원 (예측할 값의 수, 보통 1)
        dropout: 드롭아웃 확률
        """
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.future_seq = future_seq
        # 입력 피처를 d_model 차원으로 선형 변환
        self.input_linear = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        # 예측을 위한 최종 FC 레이어
        self.fc = nn.Linear(d_model, num_classes * future_seq)

    def forward(self, src):
        """
        src: [batch_size, seq_length, input_size]
        """
        # 입력 선형 변환 및 스케일 조정
        src = self.input_linear(src) * math.sqrt(self.d_model)
        # Positional Encoding 추가
        src = self.pos_encoder(src)
        # nn.TransformerEncoder는 입력 shape이 [seq_length, batch_size, d_model]이어야 함
        src = src.transpose(0, 1)  # shape: [seq_length, batch_size, d_model]
        output = self.transformer_encoder(src)
        # 마지막 시점의 출력을 사용하여 예측 (또는 필요에 따라 전체 시퀀스를 활용)
        output = output[-1, :, :]  # shape: [batch_size, d_model]
        output = self.fc(output)   # shape: [batch_size, num_classes]
         # many-to-many 형태로 reshape (예: [batch_size, 15, 1])
        output = output.view(-1, self.future_seq, num_classes)
        return output

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


# 모델, 손실함수, 옵티마이저 생성
model = TransformerModel(input_size, d_model, nhead, num_layers, num_classes, future_seq, dropout)
model = model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Setup EarlyStopping
early_stopping = EarlyStopping(patience=early_stopping_patience, delta=early_stopping_delta)

# Training Loop with EarlyStopping
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(trainX)  # outputs shape: [batch, future_seq, num_classes]
    loss = criterion(outputs, trainY)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    # Compute validation loss on test set
    model.eval()
    with torch.no_grad():
        val_outputs = model(testX)
        val_loss = criterion(val_outputs, testY)
    model.train()  # switch back to training mode

    early_stopping(val_loss.item())

    if epoch % 200 == 0:
        print(f"Epoch: {epoch}, training loss: {loss.item():.8f}, validation loss: {val_loss.item():.8f}")
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch}")
        break

print("Learning finished!")

# Testing the model
model.eval()
test_predict = model(testX)  # shape: [test_samples, future_seq, num_classes]

# Next future prediction using the latest time window
# 최근 seq_length일 데이터를 입력으로 하여 앞으로 future_seq일 치 예측
last_window_index = seq_length + future_seq
last_window = xy[-last_window_index:]  # shape: [seq_length, input_size]
last_window_tensor = torch.Tensor(last_window).unsqueeze(0).to(device)  # add batch dimension

model.eval()
with torch.no_grad():
    next_pred = model(last_window_tensor)  # shape: [1, future_seq, num_classes]

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
plt.savefig("./stock_prediction_torch_transformer.png")