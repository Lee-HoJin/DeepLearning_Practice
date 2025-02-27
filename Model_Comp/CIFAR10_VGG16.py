import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import torchvision.models.vgg as vgg

import json

import visdom
vis = visdom.Visdom()
vis.close(env="main")

history = {
    "epoch": [],
    "train_acc": [],
    "val_acc": [],
    "train_loss": [],
    "val_loss": [],
    "epoch_time": [] 
}

def value_tracker(value_plot, value, num):
    '''num, loss_value, are Tensor'''
    vis.line(X=num,
             Y=value,
             win=value_plot,
             update='append')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(777)
if device =='cuda':
    torch.cuda.manual_seed_all(777)

# 데이터 전처리: CIFAR-10
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(
    root='./cifar10',
    train=True,
    download=True,
    transform=transform)

print(trainset.data.shape)

train_data_mean = trainset.data.mean(axis=(0,1,2))
train_data_std = trainset.data.std(axis=(0,1,2))

print(train_data_mean)
print(train_data_std)

train_data_mean = train_data_mean / 255
train_data_std = train_data_std / 255

print(train_data_mean)
print(train_data_std)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(train_data_mean, train_data_std)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(train_data_mean, train_data_std)
])

trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                        download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                       download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


cfg = [32,32,'M', 64,64,128,128,128,'M',256,256,256,512,512,512,'M'] #13 + 3 =vgg16

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        #self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

vgg16= VGG(vgg.make_layers(cfg),10,True).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(vgg16.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

loss_plt = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='loss_tracker', legend=['loss'], showlegend=True))
acc_plt = vis.line(Y=torch.Tensor(1).zero_(), opts=dict(title='Accuracy', legend=['Acc'], showlegend=True))

def acc_check(net, test_set, epoch, save=1):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_set:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = (100 * correct / total)
    print('Accuracy on test images: %d %%' % acc)
    if save:
        torch.save(net.state_dict(), "./model/vit_epoch_{}_acc_{}.pth".format(epoch, int(acc)))
    return acc

print("Train Loader Length: ", len(trainloader))
epochs = 30

for epoch in range(epochs):  # 여러 epoch 동안 학습
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()  # GPU 시간 측정 시작
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = vgg16(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 30 == 29:
            value_tracker(loss_plt, torch.Tensor([running_loss / 30]),
                          torch.Tensor([i + epoch * len(trainloader)]))
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 30))
            history["train_loss"].append(running_loss)
            running_loss = 0.0
    lr_sche.step()
    
    history["epoch"].append(epoch)
    
    # Accuracy 체크
    acc = acc_check(vgg16, testloader, epoch, save=0)
    history["train_acc"].append(acc)
    value_tracker(acc_plt, torch.Tensor([acc]), torch.Tensor([epoch]))
    
    end_event.record()  # GPU 시간 측정 종료
    torch.cuda.synchronize()  # GPU 연산 동기화 (정확한 시간 측정)

    epoch_duration = start_event.elapsed_time(end_event)
    history["epoch_time"].append(epoch_duration)
    
print('Finished Training')

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = vgg16(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Final Accuracy: %d %%' % (100 * correct / total))

with open("VGG16_CIFAR10.json", "w") as f:
    json.dump(history, f)