from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Gym 환경 초기화
env = gym.make('FrozenLake-v1', is_slippery=False)  # v0 -> v1로 변경됨
num_states = env.observation_space.n
num_actions = env.action_space.n

# 신경망 모델 정의
class QNetwork(tf.keras.Model):
    def __init__(self, num_states, num_actions):
        super(QNetwork, self).__init__()
        self.dense = tf.keras.layers.Dense(num_actions, activation=None,
                                           kernel_initializer=tf.keras.initializers.RandomUniform(0, 0.01))

    def call(self, state):
        return self.dense(state)

# 모델 및 최적화 함수 설정
model = QNetwork(num_states, num_actions)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

# 학습 파라미터 설정
y = 0.99  # 할인율
e = 0.1  # 탐색 확률
num_episodes = 2000  # 에피소드 수

# 보상 및 단계 기록용 리스트
jList = []
rList = []

# 학습 루프
for i in range(num_episodes):
    s, _ = env.reset()  # 최신 Gym API에 맞게 변경
    rAll = 0
    d = False
    j = 0

    while j < 99:
        j += 1
        state_input = np.eye(num_states)[s:s+1].astype(np.float32)  # 원핫 인코딩
        
        # Epsilon-greedy 방식으로 행동 선택
        Q_values = model(state_input)
        a = np.argmax(Q_values.numpy())  # 행동 예측
        if np.random.rand(1) < e:
            a = env.action_space.sample()  # 랜덤 액션 선택
        
        # 환경에서 한 스텝 진행
        s1, r, d, _, _ = env.step(a)

        # Q-learning 업데이트 수행
        state_input_next = np.eye(num_states)[s1:s1+1].astype(np.float32)
        Q1 = model(state_input_next)
        maxQ1 = np.max(Q1.numpy())
        
        targetQ = Q_values.numpy()
        targetQ[0, a] = r + y * maxQ1

        # 손실 함수 및 최적화 수행
        with tf.GradientTape() as tape:
            Q_pred = model(state_input)
            loss = tf.reduce_sum(tf.square(targetQ - Q_pred))
        
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
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
plt.show()