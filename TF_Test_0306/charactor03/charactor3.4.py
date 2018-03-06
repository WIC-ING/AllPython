import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf
from numpy.random import RandomState
# 定义训练数据batch的大小
batch_size = 8

# 定义神经网络的参数，这里还是沿用3.4.2小节中给出的神经网络结构
w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=11))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=12))

# 在shap的一个维度上使用None可以方便使用不同的batch大小。
# 在训练时需要把数据分成比较小的batch，但时在测试时，可以一次性使用全部的数据。
# 当数据集比较小时这样比较方便测试，但数据集比较大时，将大量数据放入一个batch可能会导致内存溢出。
x  = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')


# 定义神经网络前向传播的过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)


# 定义损失函数和反向传播的算法
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdadeltaOptimizer(0.001).minimize(cross_entropy)


# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)

Y = [[int(x1+x2<1)] for (x1, x2) in X ]



with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    # 初始化变量
    sess.run(init_op)
    print(sess.run(w1))
    print(sess.run(w2))

    """
    
    """