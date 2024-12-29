import tensorflow as tf
import numpy as np

# Seed for reproducibility
tf.random.set_seed(777)

# Data
x1_data = np.array([73., 93., 89., 96., 73.], dtype=np.float32)
x2_data = np.array([80., 88., 91., 98., 66.], dtype=np.float32)
x3_data = np.array([75., 93., 90., 100., 70.], dtype=np.float32)
y_data = np.array([152., 185., 180., 196., 142.], dtype=np.float32)

# Stack data into a single array for easier manipulation
X_data = np.stack([x1_data, x2_data, x3_data], axis=1)

# Model Variables
W = tf.Variable(tf.random.normal([3, 1]), name='weights')  # weight for each feature
b = tf.Variable(tf.random.normal([1]), name='bias')  # bias term

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
        y_pred = model(X_data)
        # Compute loss
        loss = compute_loss(y_pred, y_data)

    # Compute gradients
    gradients = tape.gradient(loss, [W, b])

    # Update weights and bias
    optimizer.apply_gradients(zip(gradients, [W, b]))

    if step % 100 == 0:
        print(f"Step {step}, Cost: {loss.numpy()}, \nPrediction: \n{y_pred.numpy()}")
