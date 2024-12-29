import tensorflow as tf
import numpy as np

# Seed for reproducibility
tf.random.set_seed(777)

xy = np.loadtxt('data-01-test-score.csv',
                delimiter = ',',
                dtype = np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# Make sure the shape and data are OK
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)

# Model Variables
W = tf.Variable(tf.random.normal([x_data.shape[1], 1]), name='weights')  # Weight for each feature
b = tf.Variable(tf.random.normal([1]), name='bias')  # Bias term

# Model
def model(X):
    return tf.matmul(X, W) + b

# Loss function (Mean Squared Error)
def compute_loss(y_pred, y_true):
    return tf.reduce_mean(tf.square(y_pred - y_true))

# Optimizer
optimizer = tf.optimizers.SGD(learning_rate=1e-5)

# Training loop
for step in range(2001):
    with tf.GradientTape() as tape:
        # Make prediction
        y_pred = model(x_data)
        # Compute loss
        loss = compute_loss(y_pred, y_data)

    # Compute gradients
    gradients = tape.gradient(loss, [W, b])

    # Update weights and bias
    optimizer.apply_gradients(zip(gradients, [W, b]))

    if step == 1990 or step == 2000:
        print(f"Step {step}\nCost: {loss.numpy()}\nPrediction: \n{y_pred.numpy()}")

# 예측 함수 정의
def predict(X_input):
    return model(X_input)

# 예측 수행
print("Your score will be ", predict(tf.convert_to_tensor([[100, 70, 101]], dtype=tf.float32)).numpy())

print("Other scores will be ", predict(tf.convert_to_tensor([[60, 70, 110], [90, 100, 80]], dtype=tf.float32)).numpy())
