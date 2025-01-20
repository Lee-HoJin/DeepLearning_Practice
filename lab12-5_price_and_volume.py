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
import numpy as np
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
num_epochs = 3000
input_size = 4
output_size = 2
hidden_size = 256
timesteps = seq_length = 7
num_layers = 2  # number of layers in RNN

# Open, High, Low, Volume, Close
xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1]  # reverse order (chronically ordered)
xy = MinMaxScaler(xy)
x = xy[:, :-1]
y = xy[:, [3,4]]  # Close as label

# build a dataset
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]  # 입력 시퀀스
    _y = y[i:i + seq_length]  # 출력 시퀀스 (같은 길이)
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

# train/test split
train_size = int(len(dataY) * 0.8)
test_size = len(dataY) - train_size
trainX = torch.Tensor(np.array(dataX[0:train_size])).to(device)
testX = torch.Tensor(np.array(dataX[train_size:len(dataX)])).to(device)
trainY = torch.Tensor(np.array(dataY[0:train_size])).to(device)
testY = torch.Tensor(np.array(dataY[train_size:len(dataY)])).to(device)


class ManyToManyLSTM(nn.Module) :
    def __init__(self, input_size, hidden_size, num_layers, output_size) :
        super(ManyToManyLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 정의
        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = True)
        
        # FC 정의
        self.fc = nn.Linear(hidden_size, output_size) # output_size = 2 (Close, Volume)

    def forward(self, x) :
        # 초기 hidden state와 cell state 정의
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # LSTM Forward
        out, _ = self.lstm(x, (h_0, c_0))   # out: [batch_size, seq_length, hidden_size]

        # FC를 통해 출력으로 변환
        out = self.fc(out)

        return out

# Instantiate RNN model
model = ManyToManyLSTM(input_size, hidden_size, num_layers, output_size).to(device)

# Set loss and optimizer function
criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             weight_decay = 1e-4
                             )

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

print("Learning finished!")

# Test the model
model.eval()
with torch.no_grad():
    test_predict = model(testX)  # 예측 값
    test_predict = test_predict.cpu().numpy()  # numpy 배열로 변환
    testY = testY.cpu().numpy()  # 실제 값

# Plot predictions
plt.plot(testY[:, :, 0].flatten(), label="Actual Close Price")
plt.plot(test_predict[:, :, 0].flatten(), label="Predicted Close Price")
plt.plot(testY[:, :, 1].flatten(), label="Actual Volume")
plt.plot(test_predict[:, :, 1].flatten(), label="Predicted Volume")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Values")
# plt.show()
plt.savefig('./stock_prediction_2layers.png')