import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# Model Params
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([-0.3], tf.float32)

# Model input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

hypothesis = x * W + b

cost = tf.reduce_sum(tf.square(hypothesis - y))

train = tf.train.GradientDescentOptimizer(learning_rate = 0.01).minimize(cost)

# Training
with tf.Session() as sess :
    sess.run(tf.global_variables_initializer())

    for step in range(1000):
        sess.run(train, {x: x_train, y: y_train})

    # evaluate training accuracy
    W_val, b_val, cost_val = sess.run(
        [W, b, cost], feed_dict = {x: x_train, y: y_train}
    )

    print(f"W: {W_val} b: {b_val} cost: {cost_val}")



