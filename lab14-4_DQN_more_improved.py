import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
torch.cuda.init()
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

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print(f"\nUsing {device} device")
print("GPU: ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# 환경 설정
env = gym.make('CartPole-v2')

# 하이퍼파라미터 설정
max_episodes = 4000
input_size = env.observation_space.shape[0]  # CartPole 상태 (4)
output_size = env.action_space.n             # 행동 (2)
dis = 0.99                                   # 할인율
REPLAY_MEMORY = 30000
batch_size = 128                             # 미니배치 크기
alpha = 0.2                                  # Q-learning 업데이트 가중치
#tau = 0.005                                 # Target DQN soft update 비율
tau = 1                                      # Target DQN soft update 비율
min_buffer_size = 5000                       # 최소 Replay Buffer 크기
epsilon_decay = 0.999                        # Epsilon 지수 감소율
final_epsilon = 0.001                        # 학습 후반부에는 거의 greedy 정책 사용

learning_rate = 1e-3
hidden_size = 128

# DQN 신경망 정의
class DQN(nn.Module):
    def __init__(self, input_size, output_size, h_size=64, lr=0.001):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size)
        self.fc3 = nn.Linear(h_size, h_size)
        self.fc4 = nn.Linear(h_size, output_size)

        # 가중치 초기화
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = torch.tanh(self.fc1(x)) 
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return self.fc4(x)

    def predict(self, state):
        state = torch.tensor(state, dtype=torch.float32, device = device).unsqueeze(0)
        with torch.no_grad():
            return self.forward(state).cpu().numpy()

    def update(self, x_stack, y_stack):
        x_stack = torch.tensor(x_stack, dtype=torch.float32, device = device)
        y_stack = torch.tensor(y_stack, dtype=torch.float32, device = device)
        self.optimizer.zero_grad()
        loss = self.loss_fn(self.forward(x_stack), y_stack)
        loss.backward()
        self.optimizer.step()
        return loss.item()

# 경험 재사용 학습
def simple_replay_train(mainDQN, targetDQN, train_batch):
    x_stack = np.empty((0, mainDQN.fc1.in_features))   # 입력 (state)
    y_stack = np.empty((0, mainDQN.fc4.out_features))  # 정답 Q값 (target Q-values)

    for state, action, reward, next_state, done in train_batch:  # 경험 샘플(batch) 가져오기
        Q = mainDQN.predict(state)  # 현재 상태 s에서의 Q값 예측

        if done:
            Q[0, action] = reward  # 게임이 끝났으면 보상만 반영
        else:
            maxQ1 = np.max(targetDQN.predict(next_state))  # 다음 상태 s'에서의 최대 Q값
            Q[0, action] = (1 - alpha) * Q[0, action] + alpha * (reward + dis * maxQ1)  # Soft update 적용

        x_stack = np.vstack([x_stack, state])  # 입력 데이터 (state) 저장
        y_stack = np.vstack([y_stack, Q])      # 업데이트된 Q값 저장

    return mainDQN.update(x_stack, y_stack)  # 신경망 학습

# Target 네트워크 Soft Update
def soft_update_target(mainDQN, targetDQN, tau):
    for main_param, target_param in zip(mainDQN.parameters(), targetDQN.parameters()):
        target_param.data.copy_(tau * main_param.data + (1 - tau) * target_param.data)

# 학습 루프
def main():
    replay_buffer = deque(maxlen=REPLAY_MEMORY // 2) # 최신 데이터 중심으로 유지

    mainDQN = DQN(input_size, output_size, h_size = hidden_size, lr = learning_rate).to(device)
    targetDQN = DQN(input_size, output_size, h_size = hidden_size, lr = learning_rate).to(device)
    targetDQN.load_state_dict(mainDQN.state_dict())

    epsilon = 1.0  # 초기 epsilon 값 설정

    steps_list = []  # 각 에피소드에서의 steps를 저장할 리스트
    loss_list = []
    
    FLAG = False
    
    for episode in range(max_episodes):
        # epsilon = max(0.01, epsilon * epsilon_decay)  # Epsilon 지수 감소 적용
        epsilon = max(final_epsilon, epsilon * epsilon_decay)  # epsilon 감소
        done = False
        step_count = 0

        state = env.reset()

        while not done:
            if np.random.rand(1) < epsilon :
                action = env.action_space.sample()
            else:
                action = np.argmax(mainDQN.predict(state))

            next_state, reward, done, _ = env.step(action)

            if done:
                reward = -1

            replay_buffer.append((state, action, reward, next_state, done))

            state = next_state
            step_count += 1
            if step_count > 10000:
                break

        steps_list.append(step_count)  # steps 저장
        print(f"Episode: {episode + 1} steps: {step_count}")

        # 경험이 충분히 쌓일 때까지 학습하지 않음
        if len(replay_buffer) > min_buffer_size and episode % 20 == 1 :
            # print(f"Training at episode {episode + 1}...")
            for i in range(40):
                minibatch = random.sample(replay_buffer, batch_size)
                loss = simple_replay_train(mainDQN, targetDQN, minibatch)
                print(f"Batch {i+1}/40 - Loss: {loss:.8f}")
                
                if episode > 2000: 
                    loss_list.append(loss)

            soft_update_target(mainDQN, targetDQN, tau)

    # 학습 종료 후 그래프 그리기
    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(steps_list, label='Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps per Episode Over Training(DQN)')
    plt.legend()
    # plt.show()
    plt.savefig("Steps_plotted_DQN.png")
    plt.close()
    
    plt.figure(figsize=(10, 5), dpi=150)
    plt.plot(loss_list, label='Loss Trend')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Loss Trends Over Training(DQN)')
    plt.legend()
    # plt.show()
    plt.savefig("Loss_plotted_DQN.png")

if __name__ == "__main__":
    main()
