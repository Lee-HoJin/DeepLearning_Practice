'''
This script shows how to predict stock prices using a basic RNN
'''
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
torch.cuda.init()
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
import numpy as np
from sklearn.metrics import r2_score

import os
import matplotlib

torch.manual_seed(777)  # reproducibility

if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    matplotlib.use('Agg')


def MinMaxScaler(data):
    ''' Min Max Normalization

    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]

    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]

    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html

    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)


# train Parameters
learning_rate = 0.001
num_epochs = 1000
input_size = 4
num_classes = 2
hidden_size = 256
timesteps = seq_length = 7
num_layers = 2  # number of layers in RNN

# Open, High, Low, Volume, Close
xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1]  # reverse order (chronically ordered)

xy = MinMaxScaler(xy)
x = xy[:, :-1]    # 입력 데이터 (Open, High, Low, Volume)
y = xy[:, [3,4]]  # 출력 데이터 (Volume, Close)

# x[:, 3] = np.log1p(x[:, 3])

# build a dataset
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]   # 입력 시퀀스
    _y = y[i + seq_length - 1] # 마지막 타임스텝에 해당하는 값 (Close와 Volume)
    # print(_x, "->", _y)
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
# device = 'cpu'
print(f"\nUsing {device} device")
print("GPU: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

class EarlyStopping :
    def __init__(self, patience = 5, delta = 0) :
        self.patience = patience # 개선되지 않은 epoch 수
        self.delta = delta # 개선 기준 (손실 감소량)
        self.best_loss = None
        self.counter = 0 # 개선되지 않은 epoch 수
        self.early_stop = False

    def __call__(self, val_loss, model) :
        if self.best_loss is None :
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.delta :
            self.best_loss = val_loss
            self.counter = 0 # 성능이 개선되었으므로 카운터 리셋
        else :
            self.counter += 1

        if self.counter >= self.patience :
            self.early_stop = True
            # print(f"Early stopping triggered after {self.patience} epochs without improvement.")

        return self.early_stop


# train/test split
train_size = int(len(dataX) * 0.8)
val_size = int(0.1 * len(dataX))
test_size = len(dataX) - train_size - val_size

# # 데이터셋을 훈련, 검증, 테스트로 분리
# train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

trainX = torch.Tensor(np.array(dataX[0:train_size])).to(device)
valX = torch.Tensor(np.array(dataX[train_size  : train_size + val_size])).to(device)
testX = torch.Tensor(np.array(dataX[train_size + val_size : len(dataX)])).to(device)

trainY = torch.Tensor(np.array(dataY[0:train_size])).to(device)
valY = torch.Tensor(np.array(dataY[train_size  : train_size + val_size])).to(device)
testY = torch.Tensor(np.array(dataY[train_size + val_size : len(dataY)])).to(device)


class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        # LSTM 정의
        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True)
        
        # FC 정의
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x) :
        # 초기 hidden state와 cell state 정의
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device))
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device))

        # Propagate input thru LSTM
        _, (h_out, _) = self.lstm(x, (h_0, c_0))

        # h_out의 shape: [num_layers, batch_size, hidden_size]
        # multi-layer일 때 마지막 레이어만 사용
        h_out = h_out[-1]  # shape: [batch_size, hidden_size]

        # FC를 통해 예측값으로 변환
        out = self.fc(h_out)  # => shape: [batch_size, num_classes]

        return out
    

def init_weights(m):
    if isinstance(m, nn.Linear):
        # He 초기화
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# Instantiate RNN model
model = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length=seq_length)
model.apply(init_weights)
model = model.to(device)

# Set loss and optimizer function
criterion = torch.nn.MSELoss().to(device)    # mean-squared error for regression
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             )

# early_stopping = EarlyStopping(patience = 40, delta = 0.05)

# Train the model
for epoch in range(num_epochs):
    model.train()
    outputs = model(trainX)
    optimizer.zero_grad()
    # obtain the loss function
    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()
    # print("Epoch: %d, loss: %1.5f" % (epoch, loss.data.item()))

    # 검증 정확도 계산(Early Stopping)
    model.eval()
    val_loss = 0.0

    with torch.no_grad() :
        outputs = model(valX)
        loss = criterion(outputs, valY)
        val_loss += loss.item()

    if epoch % 20 == 0:
        print(f"Validation Loss: {val_loss:.10f},")

    # # EarlyStopping 체크
    # if early_stopping(val_loss, model):
    #     print(f"Stopping early at epoch {epoch+1}")
    #     break

print("Learning finished!")

# Test the model
model.eval()

with torch.no_grad():
    test_predict = model(testX).cpu().numpy()   # 예측 값
    test_actual  = testY.cpu().numpy() 


# (1) MAPE(Mean Absolute Percentage Error)
#     값이 0에 가까운 경우가 많으면 MAPE가 무한대로 갈 수도 있으니 주의 필요
def MAPE(true, pred):
    return np.mean(np.abs((true - pred) / (true + 1e-7))) * 100

# 실제, 예측을 각 컬럼별로 분리(0: Volume, 1: Close)
test_volume_actual = test_actual[:, 0]
test_close_actual  = test_actual[:, 1]

test_volume_pred = test_predict[:, 0]
test_close_pred  = test_predict[:, 1]

# Volume MAPE
volume_mape = MAPE(test_volume_actual, test_volume_pred)
# Close MAPE
close_mape = MAPE(test_close_actual, test_close_pred)

print(f"Volume MAPE : {volume_mape:.4f}%")
print(f"Close  MAPE : {close_mape:.4f}%")

# (2) R² Score(결정계수)
volume_r2 = r2_score(test_volume_actual, test_volume_pred)
close_r2  = r2_score(test_close_actual, test_close_pred)

print(f"Volume R2   : {volume_r2:.4f}")
print(f"Close  R2   : {close_r2:.4f}")

# Plot predictions
plt.plot(testY[:, 0], label="Actual Close Price")  # 첫 번째 열: 실제 Close Price
plt.plot(test_predict[:, 0], label="Predicted Close Price")  # 첫 번째 열: 예측된 Close Price
plt.plot(testY[:, 1], label="Actual Volume")  # 두 번째 열: 실제 Volume
plt.plot(test_predict[:, 1], label="Predicted Volume")  # 두 번째 열: 예측된 Volume
plt.legend()
plt.xlabel("Time")
plt.ylabel("Values")
# plt.show()
plt.savefig('./stock_prediction_2layers.png')