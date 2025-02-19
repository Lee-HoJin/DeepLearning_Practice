import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gym

# Gym 환경 설정
env = gym.make('CartPole-v0')

# Constants defining our neural network
input_size = env.observation_space.shape[0]  # 4
output_size = env.action_space.n  # 2

dis = 0.9  # Discount factor
REPLAY_MEMORY = 50000  # Replay memory 크기

# PyTorch 버전 DQN 클래스
class DQN(nn.Module):
    def __init__(self, input_size, output_size, h_size=20, lr=1e-2):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, h_size)
        self.fc2 = nn.Linear(h_size, output_size)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)

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

# Target DQN을 사용한 학습 함수
def simple_replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty((0, mainDQN.fc1.in_features))
    y_stack = np.empty((0, mainDQN.fc2.out_features))

    for state, action, reward, next_state, done in train_batch:
        Q = mainDQN.predict(state)

        if done:
            Q[0, action] = reward
        else:
            # Target DQN을 사용하여 안정적인 학습
            Q[0, action] = reward + dis * np.max(targetDQN.predict(next_state))

        x_stack = np.vstack([x_stack, state])
        y_stack = np.vstack([y_stack, Q])

    return mainDQN.update(x_stack, y_stack)

# 학습된 모델 테스트
def bot_replay(mainDQN):
    s = env.reset()
    reward_sum = 0
    step_count = 0
    max_steps = 500  # 최대 실행 스텝 제한

    while step_count < max_steps:
        # GUI가 없는 환경에서는 mode='rgb_array' 사용
        env.render(mode='rgb_array')  

        a = np.argmax(mainDQN.predict(s))
        s, reward, done, _ = env.step(a)
        reward_sum += reward
        step_count += 1

        if done:
            print("Total score: {}".format(reward_sum))
            break

    env.close()  # 환경 종료

# 메인 학습 함수
def main():
    max_episodes = 5000
    update_target_freq = 10 # 10 에피소드마다 target network 업데이트

    # Replay memory
    replay_buffer = deque(maxlen=REPLAY_MEMORY)

    mainDQN = DQN(input_size, output_size)
    targetDQN = DQN(input_size, output_size) # 추가된 Target Network
    targetDQN.load_state_dict(mainDQN.state_dict()) # 초기 가중치 복사

    for episode in range(max_episodes):
        e = 1. / ((episode / 10) + 1)  # Epsilon-greedy 방식으로 탐색 비율 조정
        done = False
        step_count = 0

        state = env.reset()

        while not done:
            if np.random.rand(1) < e:
                action = env.action_space.sample()  # 랜덤 행동
            else:
                action = np.argmax(mainDQN.predict(state))  # Q-network 행동 선택

            next_state, reward, done, _ = env.step(action)

            if done:  # 실패하면 보상 패널티
                reward = -100

            replay_buffer.append((state, action, reward, next_state, done))

            state = next_state
            step_count += 1
            if step_count > 10000:
                break

        print("Episode: {} steps: {}".format(episode, step_count))
        if step_count > 10000:
            pass

        if episode % 10 == 1:  # 10 에피소드마다 학습
            for _ in range(10):
                minibatch = random.sample(replay_buffer, min(len(replay_buffer), 10))
                loss = simple_replay_train(mainDQN, targetDQN,minibatch)
                print(f"Batch {i+1}/10 - Loss: {loss:.4f}")
            # print("Loss: ", loss)

         # Target 네트워크 업데이트 (10 에피소드마다)
        if episode % update_target_freq == 0:
            targetDQN.load_state_dict(mainDQN.state_dict())
            print("Target network updated!")

    bot_replay(mainDQN)

if __name__ == "__main__":
    main()
