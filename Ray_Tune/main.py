from sklearn import datasets
from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import torch
from torch import nn

from ray import tune, air
from ray.air import session

# 데이터 로드 및 전처리
data = datasets.load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=0)

# 수정 1: y_train을 float()로 변환 (회귀 문제이므로)
train_set = TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).float())
train_loader = DataLoader(train_set, batch_size=128)

x_test = torch.tensor(x_test).float()
y_test = torch.tensor(y_test).float()

def get_optimizer(model, lr):
    return torch.optim.SGD(model.parameters(), lr=lr)

def get_loss_fn():
    return torch.nn.MSELoss()

class MyModel(nn.Module):
    def __init__(self, l2_in, l2_out):
        super().__init__()
        
        self._model = nn.Sequential(
            nn.Linear(10, l2_in),
            nn.ReLU(),
            nn.Linear(l2_in, l2_out),
            nn.ReLU(),
            nn.Linear(l2_out, 1)
        )

    def forward(self, x):
        pred = self._model(x)
        # 수정 2: 차원 맞추기 - (batch_size, 1) -> (batch_size,)
        return pred.squeeze()

# 수정 3: Ray Tune에 맞게 훈련 함수 수정
def train_one_epoch(loader, model, optimizer, loss_fn):
    total_loss = 0.0
    num_batches = 0
    
    for x, y in loader:
        pred = model(x)
        loss = loss_fn(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

# 수정 4: Ray Tune에 적합한 trainable 함수
def trainable_regression(config):
    model = MyModel(config['l2_in'], config['l2_out'])
    optimizer = get_optimizer(model, lr=config['lr'])
    loss_fn = get_loss_fn()
    
    # Ray Tune의 체크포인팅을 위해 반복적으로 훈련
    for epoch in range(1000):  # 각 iteration마다 여러 에포크 훈련
        # 한 에포크 훈련
        train_loss = train_one_epoch(train_loader, model, optimizer, loss_fn)
        
        # 검증 손실 계산
        model.eval()
        with torch.no_grad():
            val_pred = model(x_test)
            val_loss = loss_fn(val_pred, y_test)
        model.train()
        
        # Ray Tune에 결과 보고 (10 에포크마다)
        if epoch % 10 == 0:
            session.report({
                "loss": float(val_loss),
                "train_loss": float(train_loss),
                "epoch": epoch
            })

# 수정 5: 적절한 하이퍼파라미터 범위 설정
param_space = {
    "l2_in": tune.grid_search([32, 64, 128]),
    "l2_out": tune.grid_search([16, 32, 64]),
    "lr": tune.uniform(1e-4, 1e-1)  # 더 적절한 learning rate 범위
}

tune_config = tune.TuneConfig(
    metric="loss",
    mode="min",
    num_samples=3
)

run_config = air.RunConfig(
    stop={"epoch": 100}  # 100 에포크 후 중지
)

tuner = tune.Tuner(
    trainable_regression,
    tune_config=tune_config,
    run_config=run_config,
    param_space=param_space,
)

if __name__ == "__main__":
    results = tuner.fit()
    print("Best config is:", results.get_best_result().config)
    print("Best loss is:", results.get_best_result().metrics["loss"])