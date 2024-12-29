# Lab 5 Logistic Regression Classifier
import tensorflow as tf
import numpy as np

# Seed for reproducibility
tf.random.set_seed(777)

# Training data
x_data = np.array([[1, 2],
                   [2, 3],
                   [3, 1],
                   [4, 3],
                   [5, 3],
                   [6, 2]], dtype=np.float32)

y_data = np.array([[0],
                   [0],
                   [0],
                   [1],
                   [1],
                   [1]], dtype=np.float32)

#### Old Version Code ####
"""
# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32) <<<=== dtype 부분은 0 또는 1로 바꿔주는 역할
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)
"""

# Define the model using tf.keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_dim=2, activation='sigmoid')
])

# Define loss function and optimizer
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
              loss='binary_crossentropy', ## 이진 교차 엔트로피, 식은 Logistic Cost Function과 같음
              metrics=['accuracy'])

# Train the model
model.fit(x_data, y_data, epochs=10000, verbose=200)

# Evaluate the model
loss, accuracy = model.evaluate(x_data, y_data, verbose=0)
print(f"Accuracy: {accuracy}")

# Predict
predictions = model.predict(x_data)
print("\nPredictions: ", predictions)