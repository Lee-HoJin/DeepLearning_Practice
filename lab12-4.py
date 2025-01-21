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
learning_rate = 0.0005
num_epochs = 3001
input_size = 5
hidden_size = 512
num_classes = 1
timesteps = seq_length = 7
num_layers = 1  # number of layers in RNN

# Open, High, Low, Volume, Close
xy = np.loadtxt('data-02-stock_daily.csv', delimiter=',')
xy = xy[::-1]  # reverse order (chronically ordered)
xy = MinMaxScaler(xy)
x = xy
y = xy[:, [-1]]  # Close as label

# build a dataset
dataX = []
dataY = []
for i in range(0, len(y) - seq_length):
    _x = x[i:i + seq_length]
    _y = y[i + seq_length]  # Next close price
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
train_size = int(len(dataX) * 0.8)
test_size = len(dataX) - train_size

# # 데이터셋을 훈련, 검증, 테스트로 분리
# train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

trainX = torch.Tensor(np.array(dataX[0:train_size])).to(device)
testX = torch.Tensor(np.array(dataX[train_size : len(dataX)])).to(device)

trainY = torch.Tensor(np.array(dataY[0:train_size])).to(device)
testY = torch.Tensor(np.array(dataY[train_size : len(dataY)])).to(device)


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        # Set parameters for RNN block
        # Note: batch_first=False by default.
        # When true, inputs are (batch_size, sequence_length, input_dimension)
        # instead of (sequence_length, batch_size, input_dimension)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(device)
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(device)

        # Propagate input through LSTM
        _, (h_out, _) = self.lstm(x, (h_0, c_0))
        # h_out의 shape: [num_layers, batch_size, hidden_size]
        # multi-layer 일 때는 마지막 레이어만 가져오기 => h_out[-1]
        h_out = h_out[-1]  # shape: [batch_size, hidden_size]

        out = self.fc(h_out)  # => shape: [batch_size, num_classes]
        return out

def init_weights(m) :
    if isinstance(m, nn.Linear) :
        # He 초기화
        nn.init.kaiming_normal_(m.weight, nonlinearity = 'relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Instantiate RNN model
lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
lstm.apply(init_weights)
lstm = lstm.to(device)

# Set loss and optimizer function
criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(),
                             lr=learning_rate,
                             weight_decay = 1e-4
                             )

# early_stopping = EarlyStopping(patience = 200, delta = 0.0001)

# Train the model
for epoch in range(num_epochs):
    lstm.train()
    outputs = lstm(trainX)
    optimizer.zero_grad()

    # obtain the loss function
    loss = criterion(outputs, trainY)
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0 :
        print("Epoch: %d, loss: %1.5f" % (epoch, loss.data.item()))

print("Learning finished!")

# Test the model
lstm.eval()
test_predict = lstm(testX)

# Plot predictions
test_predict = test_predict.data.cpu().numpy()
testY = testY.data.cpu().numpy()

# Plot predictions
plt.plot(testY, label="actual price")
plt.plot(test_predict, label="predicted price")
plt.xlabel("Time Period")
plt.ylabel("Stock Price")
plt.legend()
# plt.show()
plt.savefig("./stock_prediction_torch.png")