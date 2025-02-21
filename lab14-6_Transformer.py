import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math

import gym
from gym.envs.registration import register

# CartPole-v2 환경 등록 (max_episode_steps를 10000으로 설정)
register(
    id='CartPole-v2',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 10000},
    reward_threshold=10000.0,
)

device = 'cpu'
print(f"\nUsing {device} device")
print("GPU: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# 환경 및 하이퍼파라미터 설정
env = gym.make('CartPole-v2')
state_dim = env.observation_space.shape[0]  # 예: 4
num_actions = env.action_space.n             # 예: 2

discount = 0.99
REPLAY_MEMORY = 30000
batch_size = 64
alpha = 0.1
tau = 0.8
min_buffer_size = 2000
epsilon_decay = 0.999
final_epsilon = 0.001
max_episodes = 5000

# Transformer 네트워크 관련 하이퍼파라미터
num_layers = 4
learning_rate = 0.0005
seq_length = 100
d_model = 64
nhead = 8
dropout = 0.1

# Positional Encoding 모듈
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # 배치 차원 추가
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_length, d_model]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# Transformer 기반 Q-Network (DQN용)
class TransformerQNetwork(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, num_actions, dropout=0.1, learning_rate=0.0005, seq_length=100):
        super(TransformerQNetwork, self).__init__()
        self.d_model = d_model
        self.input_size = input_size
        self.seq_length = seq_length

        # 입력 선형 변환 및 위치 인코딩
        self.input_linear = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer 인코더 (batch_first=True이므로 입력 shape은 [batch, seq_length, d_model])
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 마지막 fully-connected layer: d_model -> num_actions
        self.fc = nn.Linear(d_model, num_actions)
        
        # Optimizer 및 Loss 함수 설정
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def forward(self, src):
        # src: [batch, seq_length, input_size]
        src = self.input_linear(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        # 마지막 타임스탭의 출력 사용 (shape: [batch, d_model])
        output = output[:, -1, :]
        output = self.fc(output)  # shape: [batch, num_actions]
        return output

    def predict(self, state_seq):
        # state_seq: numpy array shape (seq_length, input_size)
        state_seq = torch.tensor(state_seq, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            return self.forward(state_seq).cpu().numpy()

    def update(self, x_stack, y_stack):
        x_stack = torch.tensor(x_stack, dtype=torch.float32, device=device)
        y_stack = torch.tensor(y_stack, dtype=torch.float32, device=device)
        
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.forward(x_stack), y_stack)
        loss.backward()
        self.optimizer.step()
        return loss.item()

# 가중치 초기화 함수
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Replay Buffer에서 샘플을 가져와 학습하는 함수
def replay_train(policy_net, target_net, train_batch):
    x_stack = np.empty((0, policy_net.seq_length, policy_net.input_size))
    y_stack = np.empty((0, policy_net.fc.out_features))

    for state_seq, action, reward, next_state_seq, done in train_batch:
        Q = policy_net.predict(state_seq)

        if done:
            Q[0, action] = reward
        else:
            maxQ_next = np.max(target_net.predict(next_state_seq))
            Q[0, action] = (1 - alpha) * Q[0, action] + alpha * (reward + discount * maxQ_next)

        x_stack = np.vstack([x_stack, state_seq[np.newaxis, ...]])
        y_stack = np.vstack([y_stack, Q])

    return policy_net.update(x_stack, y_stack)

# Target 네트워크를 soft update 하는 함수
def soft_update_target(policy_net, target_net, tau):
    for policy_param, target_param in zip(policy_net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * policy_param.data + (1 - tau) * target_param.data)

def main():
    replay_buffer = deque(maxlen=REPLAY_MEMORY // 2)

    # 네트워크 초기화
    policy_net = TransformerQNetwork(state_dim, d_model, nhead, num_layers, num_actions, dropout, learning_rate, seq_length).to(device)
    policy_net.apply(init_weights)
    target_net = TransformerQNetwork(state_dim, d_model, nhead, num_layers, num_actions, dropout, learning_rate, seq_length).to(device)
    target_net.apply(init_weights)
    target_net.load_state_dict(policy_net.state_dict())

    epsilon = 1.0
    steps_list = []

    for episode in range(max_episodes):
        epsilon = max(final_epsilon, epsilon * epsilon_decay)
        done = False
        step_count = 0

        state = env.reset()
        # 초기 상태 시퀀스: 첫 상태를 seq_length번 반복
        state_seq = deque([state] * seq_length, maxlen=seq_length)

        while not done:
            current_state_seq = np.array(state_seq)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(policy_net.predict(current_state_seq))

            next_state, reward, done, _ = env.step(action)
            if done:
                reward = 0.1

            next_state_seq = state_seq.copy()
            next_state_seq.append(next_state)
            next_state_seq = np.array(next_state_seq)
            replay_buffer.append((current_state_seq, action, reward, next_state_seq, done))
            
            state_seq.append(next_state)
            step_count += 1
            if step_count > 10000:
                break

        steps_list.append(step_count)
        print(f"Episode: {episode + 1} steps: {step_count}")

        if len(replay_buffer) > min_buffer_size and episode % 10 == 1:
            for _ in range(10):
                minibatch = random.sample(replay_buffer, batch_size)
                loss = replay_train(policy_net, target_net, minibatch)
            soft_update_target(policy_net, target_net, tau)

    plt.figure(figsize=(10, 5))
    plt.plot(steps_list, label='Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps per Episode Over Training')
    plt.legend()
    plt.savefig("Steps_plotted_Transformer.png")

if __name__ == "__main__":
    main()
