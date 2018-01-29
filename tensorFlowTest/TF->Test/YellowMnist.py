#设置提示信息的级数（只显示 warning 和 Error）
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Mnist 数据集相关的常数
INPUT_NODE = 784
OUTPUT_NODE = 10

# 配置神经网络的参数
LAYER1_NODE = 500

BATCH_SIZE =100

LEARNING_RATE_BASE  = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99

# -------------------------------------
# ------------前向传播过程---------------
# -------------------------------------

# 使用tensorflow中的命名空间版本
#第一次调用时，reuse应赋值为False（），否则应为True
def inference(input_tensor, reuse=False):
    # 定义第一层神经网络的变量和前向传播过程
    with tf.variable_scope('layer1', reuse=reuse):
        weights = tf.get_variable('weights', [INPUT_NODE, LAYER1_NODE],
                                  initializer=tf.random_normal_initializer(stddev=0.1))
        biases = tf.get_variable('biases', [LAYER1_NODE],
                                 initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    # 类似地定义第二层神经网络的变量和前向传播过程
    with tf.variable_scope('layer2', reuse=reuse):
        weights = tf.get_variable('weights', [LAYER1_NODE, OUTPUT_NODE],
                                  initializer=tf.random_normal_initializer(stddev=0.1))
        biases  = tf.get_variable('biases', [OUTPUT_NODE],
                                  initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases
    return layer2



# # 未使用tensorflow中的命名空间版本
# def inference(input_tensor, avg_class, weights1, biases1,
#               weights2, biases2):
#
#     # 当没有提供滑动平均类时，直接使用参数当前的取值
#     if avg_class == None:
#         # 计算隐藏层的前向传播结果，这里使用了ReLU激活函数。
#         layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
#
#         return tf.matmul(layer1, weights2) + biases2
#
#     else:
#         layer1 = tf.nn.relu(
#             tf.matmul(input_tensor, avg_class.average(weights1)) +
#             avg_class.average(biases1)
#         )
#         return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


# 训练模型的过程
def train(mnist):
    x  = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')


    # 生成隐藏层的参数
    weights1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 生成输出层的参数
    weights2 = tf.Variable(
        tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))


    # 计算在当前参数下神经网络前向传播的结果。这里给出的用于计算滑动平均的类为None,
    # 所以函数不会使用参数的滑动平滑块
    y = inference(x, None, weights1, biases1, weights2, biases2)

    global_step = tf.Variable(0, trainable=False)

    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)

    variables_averages_op = variable_averages.apply(
        tf.trainable_variables())

    average_y = inference(
        x, variable_averages, weights1, biases1, weights2, biases2)



# 计算损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_, 1), logits=y)

    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    regularization = regularizer(weights1) + regularizer(weights2)

    loss = cross_entropy_mean + regularization

# 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)

# 使用tf.train.GradientDecentOptimizer优化算法来优化损失函数。注意这里损失函数包含了交叉熵损失和L2正则化损失
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # with tf.control_dependencies([train_step, variables_averages_op]):
    #     train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 循环的训练神经网络。
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                test_acc     = sess.run(accuracy, feed_dict=test_feed)
                print("After %d training step(s), validation accuracy using average model is %g"
                      "  test accuracy using average model is %g " % (i, validate_acc, test_acc))



            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc)))


#主程序入口
def main(argv=None):
    mnist = input_data.read_data_sets("../../datasets/MNIST_data", one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()




# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#
# print('Training data size: ', mnist.train.num_examples)
#
# print('Validating data size； ', mnist.validation.num_examples)
#
# print('Testing data size: ', mnist.test.num_examples)
#
# # print('Example training data: ', mnist.train.images[0])
# #
# # print('Example training label: ', mnist.train.labels[0])
#
# batch_size = 100
# xs, ys = mnist.train.next_batch(batch_size)
#
# print('X shape: ', xs.shape)
# print('Y shape: ', ys.shape)


