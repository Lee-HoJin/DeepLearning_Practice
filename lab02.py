import tensorflow as tf

# X and Y data
x_train = tf.constant([1, 2, 3], dtype=tf.float32)
y_train = tf.constant([1, 2, 3], dtype=tf.float32)

# Variables
W = tf.Variable(tf.random.normal([1]), name='weight')
b = tf.Variable(tf.random.normal([1]), name='bias')

# Our Hypothesis Wx + b
hypothesis = x_train * W + b # Node
## 2.0 이상 버전
def predict(X):
    return x_train * W + b

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))
## reduce_mean은 tensor의 평균 구하는 함수
## 2.0 이상 버전
def cost (y_predict, y_true):
    return tf.reduce_mean(tf.square(y_predict - y_true))

# Minimize
# optimizer = tf.optimizers.SGD(learning_rate = 0.01)
# train = optimizer.minimize(cost)

learning_rate = 0.01

# Training loop
for step in range(2001):
    with tf.GradientTape() as tape:
        # Hypothesis
        hypothesis = x_train * W + b
        
        # Cost/loss function
        cost = tf.reduce_mean(tf.square(hypothesis - y_train))
    
    # Compute gradients
    gradients = tape.gradient(cost, [W, b])
    
    # Apply gradients using SGD
    optimizer = tf.optimizers.SGD(learning_rate)
    optimizer.apply_gradients(zip(gradients, [W, b]))

    # Print step, cost, and variables every 20 steps
    continue
    if step % 20 == 0:
        print(f"Step: {step}, Cost: {cost.numpy()}, W: {W.numpy()}, b: {b.numpy()}")


