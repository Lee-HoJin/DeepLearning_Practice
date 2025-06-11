from sklearn import datasets
from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

import torch
from torch import nn

from ray import tune, air
from ray.air import session

data = datasets.load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=0)

train_set = TensorDataset(torch.tensor(x_train).float(), torch.tensor(y_train).long())
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
    return pred
  

def train(loader, model, optimizer, loss_fn):
  epoch=1000
  print_num=100

  for e in range(epoch):
    for i, (x, y) in enumerate(loader):
      pred = model(x)
      loss = loss_fn(pred, y)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      if e % print_num == 0:
        print(e, i, loss.item())
  print(e, i, loss.item())

def trainable_regression(config):
    model = MyModel(config['l2_in'], config['l2_out'])
    optimizer = get_optimizer(model, lr=config['lr'])
    loss_fn = get_loss_fn()

    while True:
        train(train_loader, model, optimizer, loss_fn)
        loss_val = loss_fn(model(x_test), y_test)
        tune.report(loss=float(loss_val))

param_space = {
    "l2_in": tune.grid_search([10, 100, 200]),
    "l2_out": tune.grid_search([10, 100, 200]),
    "lr": tune.uniform(1e-10, 1e-7)
    }

tune_config=tune.TuneConfig(
    metric="loss",
    mode="min",
    num_samples=3
)

run_config=air.RunConfig(
    stop={"training_iteration": 2}
)

tuner = tune.Tuner(
    trainable_regression,
    tune_config=tune_config,
    run_config=run_config,
    param_space=param_space,
)

results = tuner.fit()
print("Best config is:", results.get_best_result().config)