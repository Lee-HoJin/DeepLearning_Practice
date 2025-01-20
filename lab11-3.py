# Lab 11 MNIST and Deep learning CNN
import torch
torch.cuda.init()
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch.nn.init

torch.manual_seed(777)  # reproducibility

# parameters
learning_rate = 0.0001
training_epochs = 50
batch_size = 128

# MNIST dataset
mnist_train = dsets.MNIST(root='MNIST_data/',
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root='MNIST_data/',
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

# dataset loader
data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True)

# CNN Model
class CNN(torch.nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        self._build_net()

    def _build_net(self):
        # L1 ImgIn shape=(?, 28, 28, 1)
        #    Conv     -> (?, 28, 28, 32)
        #    Pool     -> (?, 14, 14, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.2))
        # L2 ImgIn shape=(?, 14, 14, 32)
        #    Conv      ->(?, 14, 14, 64)
        #    Pool      ->(?, 7, 7, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Dropout(0.2))
        # L3 ImgIn shape=(?, 7, 7, 64)
        #    Conv      ->(?, 7, 7, 128)
        #    Pool      ->(?, 4, 4, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            torch.nn.Dropout(0.2))
        # L4 FC 4x4x128 inputs -> 625 outputs
        self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5))
        # L5 Final FC 625 inputs -> 10 outputs
        self.fc2 = torch.nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

        # define cost/loss & optimizer
        # Softmax is internally computed.
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.fc1(out)
        out = self.fc2(out)
        return out

    def predict(self, x):
        self.eval()
        return self.forward(x)

    def get_accuracy(self, x, y):
        prediction = self.predict(x)
        correct_prediction = (torch.max(prediction.data, 1)[1] == y.data)
        self.accuracy = correct_prediction.float().mean()
        return self.accuracy

    def train_model(self, x, y):
        self.train()
        self.optimizer.zero_grad()
        hypothesis = self.forward(x)
        self.cost = self.criterion(hypothesis, y)
        self.cost.backward()
        self.optimizer.step()
        return self.cost


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


num_models = 5
ensemble_models = [CNN().to(device) for _ in range(num_models)]

# train my model
print('Learning started. It takes sometime.')
model_count = 0
for model in ensemble_models :
    model_count += 1
    print(f"Model {model_count}/{num_models} is now learning")
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = len(mnist_train) // batch_size

        for i, (batch_xs, batch_ys) in enumerate(data_loader):
            # image is already size of (28x28), no reshape
            X, Y = batch_xs.to(device), batch_ys.to(device)

            cost = model.train_model(X, Y)

            avg_cost += cost.data / total_batch

        # print("[Epoch: {:>4}] cost = {:>.9}".format(epoch + 1, avg_cost.item()))

print('Learning Finished!')

# Ensemble Evaluation
def ensemble_predict (models, X) :
    predictions = [model.predict(X).detach() for model in models]

    ## 결과 값의 평균 사용
    avg_prediction = torch.mean(torch.stack(predictions), dim = 0)
    return avg_prediction

    ## 결과 값의 합 사용
    # summed_prediction = torch.sum(torch.stack(predictions), dim = 0)
    # return summed_prediction

X_test = mnist_test.data.view(len(mnist_test), 1, 28, 28).float()
Y_test = mnist_test.targets

X_test, Y_test = X_test.to(device), Y_test.to(device)

ensemble_output = ensemble_predict(ensemble_models, X_test)
ensemble_accuracy = (torch.max(ensemble_output, 1)[1] == Y_test).float().mean()

print("Model 3")
print("Ensemble Accuracy:", ensemble_accuracy.item())
print("HyperParams")
print("Learning Rate: ", learning_rate)
print("Batch Size: ", batch_size)
print("Training Epochs: ", training_epochs)