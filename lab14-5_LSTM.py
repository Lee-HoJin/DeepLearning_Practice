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
from gym.envs.registration import register

# Register CartPole with user-defined max_episode_steps
register(
    id='CartPole-v2',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 10000},
    reward_threshold=10000.0,
)

# device = 'cuda'
device = 'cpu'
print(f"\nUsing {device} device")
print("GPU: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# 환경 설정
env = gym.make('CartPole-v2')

# 하이퍼파라미터 설정
max_episodes = 4000
input_size = env.observation_space.shape[0]  # 상태 차원 (4)
num_classes = env.action_space.n             # 행동 개수 (2)
dis = 0.99                                   # 할인율
REPLAY_MEMORY = 80000
batch_size = 128                              # 미니배치 크기
alpha = 0.1                                  # Q-learning 업데이트 가중치
tau = 1                                      # Target 네트워크 Soft Update 비율
min_buffer_size = 8000                       # 최소 Replay Buffer 크기
epsilon_decay = 0.999                        # Epsilon 지수 감소율
final_epsilon = 0.001                        # 학습 후반부에는 거의 greedy 정책 사용

# RNN/LSTM 관련 파라미터
num_layers = 4
hidden_size = 64
seq_length = 10  

# 전역 학습률
learning_rate = 1e-3

class LSTM(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length, learning_rate=1e-2):
        super(LSTM, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # x의 shape: [batch, seq_length, input_size]
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        _, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out[-1]  # 마지막 레이어의 은닉 상태 사용, shape: [batch, hidden_size]
        out = self.fc(h_out)  # shape: [batch, num_classes]
        return out
    
    def predict(self, state_seq):
        # state_seq: numpy array shape (seq_length, input_size)
        # 배치 차원을 추가하여 (1, seq_length, input_size)로 변환
        state_seq = torch.tensor(state_seq, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            return self.forward(state_seq).cpu().numpy()

    def update(self, x_stack, y_stack):
        # x_stack: numpy array shape (batch, seq_length, input_size)
        # y_stack: numpy array shape (batch, num_classes)
        x_stack = torch.tensor(x_stack, dtype=torch.float32, device=device)
        y_stack = torch.tensor(y_stack, dtype=torch.float32, device=device)
        
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.forward(x_stack), y_stack)
        loss.backward()
        self.optimizer.step()
        return loss.item()

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def simple_replay_train(mainLSTM, targetLSTM, train_batch):
    # x_stack: (batch, seq_length, input_size), y_stack: (batch, num_classes)
    x_stack = np.empty((0, mainLSTM.seq_length, mainLSTM.input_size))
    y_stack = np.empty((0, mainLSTM.fc.out_features))

    for state_seq, action, reward, next_state_seq, done in train_batch:
        Q = mainLSTM.predict(state_seq)

        if done:
            Q[0, action] = reward
        else:
            maxQ1 = np.max(targetLSTM.predict(next_state_seq))
            Q[0, action] = (1 - alpha) * Q[0, action] + alpha * (reward + dis * maxQ1)

        # state_seq는 이미 (seq_length, input_size) shape이므로 새로운 축을 추가해 (1, seq_length, input_size)로 맞춤
        x_stack = np.vstack([x_stack, state_seq[np.newaxis, ...]])
        y_stack = np.vstack([y_stack, Q])

    return mainLSTM.update(x_stack, y_stack)

def soft_update_target(mainLSTM, targetLSTM, tau):
    for main_param, target_param in zip(mainLSTM.parameters(), targetLSTM.parameters()):
        target_param.data.copy_(tau * main_param.data + (1 - tau) * target_param.data)

def main():    
    replay_buffer = deque(maxlen=REPLAY_MEMORY // 2)  # 최신 데이터 중심 유지

    mainLSTM = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length, learning_rate).to(device)
    mainLSTM.apply(init_weights)
    targetLSTM = LSTM(num_classes, input_size, hidden_size, num_layers, seq_length, learning_rate).to(device)
    targetLSTM.apply(init_weights)
    targetLSTM.load_state_dict(mainLSTM.state_dict())

    epsilon = 1.0
    steps_list = []
    loss_list = []
    

    for episode in range(max_episodes):
        epsilon = epsilon * epsilon_decay
        done = False
        step_count = 0

        state = env.reset()
        # 초기 상태 시퀀스: 첫 상태를 seq_length번 반복하여 채움
        state_seq = deque([state] * seq_length, maxlen=seq_length)

        if episode < 50 :
            print(state_seq)
            print("_")

        while not done:
            current_state_seq = np.array(state_seq)
            if np.random.rand() < epsilon and episode < 3000:
                action = env.action_space.sample()
            else:
                action = np.argmax(mainLSTM.predict(current_state_seq))

            next_state, reward, done, _ = env.step(action)
            if done:
                reward = 0.1

            # 다음 상태 시퀀스 생성: 기존 시퀀스에 next_state 추가 (가장 오래된 상태는 제거됨)
            next_state_seq = state_seq.copy()
            next_state_seq.append(next_state)
            next_state_seq = np.array(next_state_seq)

            replay_buffer.append((current_state_seq, action, reward, next_state_seq, done))
            
            # 현재 시퀀스 업데이트
            state_seq.append(next_state)
            step_count += 1
            if step_count > 10000:
                break

        steps_list.append(step_count)
        print(f"Episode: {episode + 1} steps: {step_count}")

        if len(replay_buffer) > min_buffer_size and episode % 20 == 1:
            for i in range(20):
                minibatch = random.sample(replay_buffer, batch_size)
                loss = simple_replay_train(mainLSTM, targetLSTM, minibatch)
                print(f"Batch {i+1}/20 - Loss: {loss:.6f}")
                
                if episode > 2000:
                    loss_list.append(loss)
                
            soft_update_target(mainLSTM, targetLSTM, tau)

    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(steps_list, label='Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps per Episode Over Training(LSTM)')
    plt.legend()
    plt.savefig("Steps_plotted_LSTM.png")
    plt.close()
    
    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(loss_list, label='Loss Trend')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss Trends Over Training(LSTM)')
    plt.legend()
    # plt.show()
    plt.savefig("Loss_plotted_LSTM.png")

if __name__ == "__main__":
    main()
