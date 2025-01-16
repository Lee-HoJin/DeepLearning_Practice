# Lab 9 XOR
import torch
torch.cuda.init()  # CUDA 초기화
from torch import nn
from torch.autograd import Variable
import numpy as np

torch.manual_seed(777)  # for reproducibility

x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)

X = Variable(torch.from_numpy(x_data))
Y = Variable(torch.from_numpy(y_data))

# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
# print(f"Using {device} device")

device = "cpu"

# 신경망 클래스 생성
class NeuralNetwork(nn.Module) :
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.layer_stack = nn.Sequential(
            nn.Linear(2, 4),
            nn.Sigmoid(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )

    def forward(self, x) :
        x = self.flatten(x)
        results = self.layer_stack(x)

        return results
    
# 모델 정의 및 훈련
model = NeuralNetwork().to(device)
criterion = nn.BCELoss
optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

for step in range(10001):
    X, Y = X.to(device), Y.to(device)

    optimizer.zero_grad()
    hypothesis = model(X)

    cost = -(Y * torch.log(hypothesis) + (1 - Y)
             * torch.log(1 - hypothesis)).mean()
    
    cost.backward()
    optimizer.step()

    if step % 500 == 0:
        print(step, cost.data.numpy())

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = (model(X).data > 0.5).float()
accuracy = (predicted == Y.data).float().mean()
print("\nHypothesis: \n", hypothesis.data.numpy(), "\nCorrect: \n",
      predicted.numpy(), "\nAccuracy: ", accuracy)
