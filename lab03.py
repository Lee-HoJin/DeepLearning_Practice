import tensorflow as tf
import matplotlib.pyplot as plt

# 데이터
X = [1, 2, 3]
Y = [1, 2, 3]

# 변수 W 정의
W = tf.Variable(0.0, dtype=tf.float32)  # 초기값을 0으로 설정

# Our Hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# 비용 함수와 W 값 기록을 위한 리스트
W_history = []
cost_history = []

# 최적화 함수 (옵티마이저)
optimizer = tf.optimizers.SGD(learning_rate=0.01)

# W 업데이트 함수
def run_optimization():
    with tf.GradientTape() as tape:
        tape.watch(W)
        cost_value = tf.reduce_mean(tf.square(X * W - Y))
    gradients = tape.gradient(cost_value, W)
    optimizer.apply_gradients([(gradients, W)])
    return cost_value

# 그래디언트 계산을 통한 W 값 업데이트
for i in range(-30, 50):
    curr_W = i * 0.1
    W.assign(curr_W)  # W 값 갱신
    curr_cost = run_optimization()  # 비용 계산
    W_history.append(curr_W)
    cost_history.append(curr_cost)

# Linear Model
def hypothesis (X, W) :
    return X * W

# cost/loss function
def cost_fn(X, Y, W) :
    return tf.reduce_mean( tf.square(hypothesis(X, W) - Y) )

# Gradient Computation
def compute_gradient(X, Y, W) :
    with tf.GradientTape() as tape:
        tape.watch(W)
        cost_value = cost_fn(X, Y,W)
    gradient = tape.gradient(cost_value, W)
    return gradient

# Create an Optimizer
optimizer = tf.optimizers.SGD(learning_rate = 0.01)

# Training Loop
for step in range(101):
    # compute the gradient
    gradient_val = compute_gradient(X, Y, W)

    # apply gradients to update weights
    optimizer.apply_gradients([(gradient_val, W)])

    # print the step, gradient value, and current weight
    print(step, gradient_val.numpy(), W.numpy())

# # 비용 함수 시각화
# plt.plot(W_history, cost_history)
# plt.xlabel("W value")
# plt.ylabel("Cost value")
# plt.title("Cost Function")
# plt.savefig("lab03.png")
