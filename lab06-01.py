# Lab 6 Softmax Classifier

import tensorflow as tf
import numpy as np

# for reproducibility
tf.random.set_seed(777)

# Data
x_data = np.array([[1, 2, 1, 1],
                   [2, 1, 3, 2],
                   [3, 1, 3, 4],
                   [4, 1, 5, 5],
                   [1, 7, 5, 5],
                   [1, 2, 5, 6],
                   [1, 6, 6, 6],
                   [1, 7, 7, 7]])

y_data = np.array([[0, 0, 1],
                   [0, 0, 1],
                   [0, 0, 1],
                   [0, 1, 0],
                   [0, 1, 0],
                   [0, 1, 0],
                   [1, 0, 0],
                   [1, 0, 0]])

# Model setup
model = tf.keras.Sequential([
    tf.keras.layers.Dense(3, input_shape=(4,), activation='softmax')
])

# Compile model with cross-entropy loss
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_data, y_data, epochs=2001, verbose=0)

    # Testing & One-hot encoding
test_data = np.array([[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]])

# Predict and print results
predictions = model.predict(test_data)
for prediction in predictions:
    print(prediction, np.argmax(prediction))

# To simulate the same behavior for multiple inputs
all_predictions = model.predict(test_data)
print(all_predictions)
print(np.argmax(all_predictions, axis=1))