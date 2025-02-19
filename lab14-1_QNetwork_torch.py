import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# PyTorch 모델 정의 (Q-network)
class QNetwork(nn.Module):
    def __init__(self, num_states, num_actions):
        super(QNetwork, self).__init__()
        self.fc = nn.Linear(num_states, num_actions)  # 단순한 하나의 선형 레이어
        nn.init.uniform_(self.fc.weight, 0, 0.01)  # 가중치 초기화

    def forward(self, x):
        return self.fc(x)

# 환경 초기화
env = gym.make('FrozenLake-v0')
num_states = env.observation_space.n  # 상태 개수 (16)
num_actions = env.action_space.n  # 행동 개수 (4)

# 모델 및 최적화 함수 설정
model = QNetwork(num_states, num_actions)
optimizer = optim.SGD(model.parameters(), lr=0.1)  # 경사 하강법 (SGD)
loss_fn = nn.MSELoss()  # 손실 함수 (평균제곱오차)

# 학습 파라미터 설정
y = 0.99  # 할인율
e = 0.1  # 탐색 확률
num_episodes = 2000  # 학습할 에피소드 수

# 보상 및 단계 기록용 리스트
jList = []
rList = []

# 학습 루프
for i in range(num_episodes):
    s = env.reset()  # Gym 0.15.4에서는 단일 값 반환
    rAll = 0
    d = False
    j = 0

    while j < 99:
        j += 1
        state_input = torch.tensor(np.identity(num_states)[s:s+1], dtype=torch.float32)

        # Epsilon-greedy 방식으로 행동 선택
        with torch.no_grad():  # 탐색이므로 그래디언트 X
            Q_values = model(state_input)
        a = torch.argmax(Q_values).item()

        if np.random.rand(1) < e:
            a = env.action_space.sample()

        # 환경에서 한 스텝 진행
        s1, r, d, _ = env.step(a)

        # Q-learning 업데이트 수행
        state_input_next = torch.tensor(np.identity(num_states)[s1:s1+1], dtype=torch.float32)
        with torch.no_grad():
            Q1 = model(state_input_next)
            maxQ1 = torch.max(Q1).item()

        targetQ = Q_values.clone().detach()
        targetQ[0, a] = r + y * maxQ1

        # 모델 업데이트 (PyTorch 방식)
        optimizer.zero_grad()
        Q_pred = model(state_input)
        loss = loss_fn(Q_pred, targetQ)  # 손실 계산
        loss.backward()
        optimizer.step()

        rAll += r
        s = s1

        if d:
            e = 1. / ((i / 50) + 10)  # 탐색 확률 감소
            break

    jList.append(j)
    rList.append(rAll)

print("Percent of successful episodes: {:.2f}%".format(100 * sum(rList) / num_episodes))

# 학습 결과 시각화
plt.plot(rList)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Reward per Episode')
# plt.show()
plt.savefig("Q_Network_FrozenLake.png")
