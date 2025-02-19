import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import gym

# 환경 설정
env = gym.make('CartPole-v0')

# 하이퍼파라미터 설정
input_size = env.observation_space.shape[0]  # CartPole 상태 (4)
output_size = env.action_space.n  # 행동 (2)
dis = 0.99  # 할인율
REPLAY_MEMORY = 50000
batch_size = 64  # 미니배치 크기
update_target_freq = 20  # Target DQN 업데이트 주기
alpha = 0.1  # Q-learning 업데이트 가중치

# DQN 신경망 정의
class DQN(nn.Module):
    def __init__(self, input_size, output_size, h_size=32, lr=0.001):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, output_size)

        # 가중치 초기화
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def predict(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            return self.forward(state).numpy()

    def update(self, x_stack, y_stack):
        x_stack = torch.tensor(x_stack, dtype=torch.float32)
        y_stack = torch.tensor(y_stack, dtype=torch.float32)
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.forward(x_stack), y_stack)
        loss.backward()
        self.optimizer.step()
        return loss.item()

# 경험 재사용 학습
def simple_replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty((0, mainDQN.fc1.in_features))
    y_stack = np.empty((0, mainDQN.fc3.out_features))

    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)

        if done:
            Q[0, action] = reward
        else:
            maxQ1 = np.max(targetDQN.predict(next_state))
            Q[0, action] = (1 - alpha) * Q[0, action] + alpha * (reward + dis * maxQ1)  # Soft update 적용

        x_stack = np.vstack([x_stack, state])
        y_stack = np.vstack([y_stack, Q])

    return mainDQN.update(x_stack, y_stack)

# 학습 루프
def main():
    max_episodes = 5000
    replay_buffer = deque(maxlen=REPLAY_MEMORY)

    mainDQN = DQN(input_size, output_size)
    targetDQN = DQN(input_size, output_size)
    targetDQN.load_state_dict(mainDQN.state_dict())

    for episode in range(max_episodes):
        e = max(0.01, 1.0 - episode / 5000)  # Epsilon 선형 감소
        done = False
        step_count = 0

        state = env.reset()

        while not done:
            if np.random.rand(1) < e:
                action = env.action_space.sample()
            else:
                action = np.argmax(mainDQN.predict(state))

            next_state, reward, done, _ = env.step(action)

            if done:
                reward = -100  # 실패하면 보상 패널티

            replay_buffer.append((state, action, reward, next_state, done))

            state = next_state
            step_count += 1
            if step_count > 10000:
                break

        print(f"Episode: {episode} steps: {step_count}")

        if len(replay_buffer) > batch_size and episode % 10 == 1:
            print(f"Training at episode {episode}...")
            for i in range(10):
                minibatch = random.sample(replay_buffer, batch_size)
                loss = simple_replay_train(mainDQN, targetDQN, minibatch)
                print(f"Batch {i+1}/10 - Loss: {loss:.4f}")

        if episode % update_target_freq == 0:
            targetDQN.load_state_dict(mainDQN.state_dict())
            print("Target network updated!")

if __name__ == "__main__":
    main()
