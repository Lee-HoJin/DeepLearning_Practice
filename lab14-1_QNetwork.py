from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# TensorFlow 2.x에서 1.x 스타일의 그래프 실행을 위해 Eager Execution 비활성화
tf.compat.v1.disable_eager_execution()

# Gym 환경 초기화
env = gym.make('FrozenLake-v0')

# 신경망 모델 정의
inputs1 = tf.compat.v1.placeholder(shape=[1,16], dtype=tf.float32)
W = tf.Variable(tf.random.uniform([16,4], 0, 0.01))
Qout = tf.matmul(inputs1, W)
predict = tf.argmax(Qout, 1)

# 손실 함수 및 최적화 설정
nextQ = tf.compat.v1.placeholder(shape=[1,4], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

# 초기화 연산 정의
init = tf.compat.v1.global_variables_initializer()

# 학습 파라미터 설정
y = 0.99  # 할인율
e = 0.1  # 탐색 확률
num_episodes = 2000  # 학습할 에피소드 수

# 보상 및 단계 기록용 리스트
jList = []
rList = []

# TensorFlow 1.x 스타일의 세션 실행
with tf.compat.v1.Session() as sess:
    sess.run(init)

    for i in range(num_episodes):
        s = env.reset()  # Gym 0.15.4에서는 단일 값 반환
        rAll = 0
        d = False
        j = 0

        while j < 99:
            j += 1
            # 현재 상태의 원핫 인코딩 벡터 생성
            state_input = np.identity(16)[s:s+1]

            # Q-network로 행동 예측
            a, allQ = sess.run([predict, Qout], feed_dict={inputs1: state_input})

            # Epsilon-greedy 정책 적용
            if np.random.rand(1) < e:
                a[0] = env.action_space.sample()

            # 환경에서 한 스텝 진행
            s1, r, d, _ = env.step(a[0])

            # Q-learning 업데이트 수행
            Q1 = sess.run(Qout, feed_dict={inputs1: np.identity(16)[s1:s1+1]})
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0, a[0]] = r + y * maxQ1

            # 모델 학습
            _, W1 = sess.run([updateModel, W], feed_dict={inputs1: state_input, nextQ: targetQ})

            rAll += r
            s = s1

            if d:
                e = 1./((i/50) + 10)  # 탐색 확률 감소
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