import torch
import torch.nn as nn
import torch.optim as optim

import json

import torchvision
import torchvision.transforms as transforms

import visdom
vis = visdom.Visdom(port=8097)
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

# Vision Transformer (ViT) 모델 정의
class VisionTransformer(nn.Module):
    def __init__(self, image_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=128, depth=6, num_heads=8, mlp_ratio=4, dropout_rate=0.1):
        super(VisionTransformer, self).__init__()
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        
        # 패치 임베딩: Conv2d를 사용해 패치를 임베딩함
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # 분류 토큰 및 위치 임베딩
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout_rate)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim,
                                                   nhead=num_heads,
                                                   dim_feedforward=embed_dim * mlp_ratio,
                                                   dropout=dropout_rate,
                                                   activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 분류 헤드
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self._init_weights()
        
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.kaiming_normal_(self.patch_embed.weight, mode='fan_out', nonlinearity='relu')
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        
    def forward(self, x):
        B = x.shape[0]
        # x: [B, 3, 32, 32]
        x = self.patch_embed(x)  # [B, embed_dim, 32/patch_size, 32/patch_size]
        x = x.flatten(2)         # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)     # [B, num_patches, embed_dim]
        
        # 클래스 토큰 추가
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B, num_patches+1, embed_dim]
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer는 [sequence_length, batch, embed_dim] 형태의 입력을 요구
        x = x.transpose(0, 1)  # [num_patches+1, B, embed_dim]
        x = self.transformer(x)
        x = x.transpose(0, 1)  # [B, num_patches+1, embed_dim]
        
        # 분류 토큰 추출 후 정규화 및 헤드 통과
        x = self.norm(x[:, 0])
        x = self.head(x)
        return x

vit_model = VisionTransformer(image_size=32, patch_size=4, in_channels=3, num_classes=10,
                              embed_dim=348, depth=12, num_heads=12, mlp_ratio=4, dropout_rate=0.1).to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(vit_model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-5)
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
        outputs = vit_model(inputs)
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
    acc = acc_check(vit_model, testloader, epoch, save=0)
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
        outputs = vit_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Final Accuracy: %d %%' % (100 * correct / total))

with open("ResNet50.json", "w") as f:
    json.dump(history, f)
