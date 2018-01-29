#设置提示信息的级数（只显示 warning 和 Error）
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# def weight_variable(shape):
#     initial = tf.truncated_normal(shape, stddev=0.1)
#     return tf.Variable(initial)
#
# def bias_variale(shape):
#     initial = tf.constant(0.1, shape=shape)
#     return tf.Variable(initial)

sess = tf.InteractiveSession()

x = tf.placeholder('float', shape=[None, 784])

y_ = tf.placeholder('float', shape=[None, 10])


W = tf.Variable(tf.zeros([784,10]))

b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

