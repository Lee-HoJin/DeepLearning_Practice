# Lab 5 Logistic Regression Classifier
import tensorflow as tf
import numpy as np

#### Old Version Code ####
"""
tf.set_random_seed(777)  # for reproducibility

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

# placeholders for a tensor that will be always fed.
X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(-tf.matmul(X, W)))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

# cost/loss function
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *
                       tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
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

# Seed for Reproductivity
tf.random.set_seed(777)

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]

# print(x_data.shape, y_data.shape)

# Define the Model Class
class LogisticRegressionModel(tf.keras.Model) :
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        # Define Model Layers
        self.dense = tf.keras.layers.Dense(
            1,
            activation = 'sigmoid',
            input_shape = (input_dim,)
        )
        
    def call(self, inputs) :
        return self.dense(inputs)
    
# Initialize the Model
input_dim = x_data.shape[1] ## Features(Column)의 개수
model = LogisticRegressionModel(input_dim)

# Compile the Model
model.compile(
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01),
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics = ['accuracy']
)

# Train the Model
model.fit(x_data, y_data,
          epochs = 10001,
          batch_size = 32,
          verbose = 200
)

# Evaluate Accuracy
h = model.predict(x_data)
predicted = (h > 0.5).astype(np.float32)
accuracy = np.mean(predicted == y_data)

print("\nHypothesis: ", h)
print("\nCorrect (Y): ", predicted)
print("\nAccuracy: ", accuracy)