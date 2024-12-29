# Lab 6 Softmax Classifier

import tensorflow as tf
import numpy as np

# for reproducibility
tf.random.set_seed(777)

# Load dataset
xy = np.loadtxt('data-04-zoo.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

# Number of classes (7 classes for animal types)
# 0부터 6까지
nb_classes = 7

# One-hot encoding for the labels
## to_categorical을 사용하여 One-Hot 인코딩 과정 최적화
y_data_one_hot = tf.keras.utils.to_categorical(y_data, nb_classes)

# Build the model using tf.keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(nb_classes, input_shape=(16,), activation='softmax')
])

# Compile the model with SGD optimizer and categorical crossentropy loss
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss='categorical_crossentropy',
              metrics=['accuracy']
)

# Train the model
model.fit(x_data, y_data_one_hot, epochs=2001, batch_size=32, verbose=0)

# Test and evaluate predictions
predictions = model.predict(x_data)

# Print results
## zip을 이용하여 두 개 이상의 객체 iteration을 동시에 할 수 있음
## flatten()은 1차원 배열로 바꿔줘서 iteration이 가능하도록 해줌 (reshape과 같음?)
for pred, true in zip(np.argmax(predictions, axis=1), y_data.flatten()):
    print(f"[{pred == int(true)}] Prediction: {pred} True Y: {int(true)}")

