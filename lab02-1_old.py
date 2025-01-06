import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

tf.set_random_seed(777)

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.random_normal([1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "bias")

# Linear Regression
hypothesis = x_train * W + b

# Cost/Loss Function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Optimzer
train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

# Launch the graph in a session
with tf.Session() as sess :
    # Initializes global variables in the graph
    sess.run(tf.global_variables_initializer())

    # Fit the line
    for step in range(2001) :
        _, cost_val, W_val, b_val = sess.run([train, cost, W, b])

        if step % 20 == 0 :
            print(step, cost_val, W_val, b_val)