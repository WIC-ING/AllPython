#设置提示信息的级数（只显示 warning 和 Error）
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
#导入TensorFlow
import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

# 两个输入节点
x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-output')


# 定义一个单层的神经网络前向传播的过程，这里是简单的加权和
w1 = tf.Variable(tf.random_normal([2,1], stddev=1, seed=1))
y  = tf.matmul(x, w1)

#定义带有正则项的损失函数
loss = tf.reduce_mean(tf.square(y_ - y)) + tf.contrib.layers.l1_regularizer(.5)(w1)
# 定义预测多了和预测少了的成本 -- 损失函数

# loss_less = 1
# loss_more = 10
# loss = tf.reduce_mean(tf.where(tf.greater(y, y_),
#                                 (y - y_) * loss_more,
#                                 (y_ - y) * loss_less
# ))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# global_step = tf.Variable(0)
#
# learning_rate = tf.train.exponential_decay(
#     0.1, global_step, 100, 0.96, staircase=True
# )
#
# learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[x1+x2+rdm.rand()/10.0-0.05] for (x1,x2) in X]

#训练神经网络
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size)%dataset_size
        end   = min(start+batch_size, dataset_size)

        sess.run(train_step,
                 feed_dict={x:X, y_:Y})
    print(sess.run(w1))
#
# batch_size = 8
#
# w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
# w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))
#
# x  = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
# y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')
#
# a = tf.matmul(x, w1)
# y = tf.matmul(a, w2)
#
# # 定义交叉熵
# cross_entropy = -tf.reduce_mean(
#     y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
# )
# train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
#
# # 通过随机数生成一个模拟数据集
# rdm = RandomState(1)
# dataset_size = 128
# X = rdm.rand(dataset_size, 2)
#
# Y = [[int(x1+x2 < 1)] for (x1, x2) in X]
#
#
# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     #初始化变量
#     sess.run(init_op)
#     print(sess.run(w1))
#     print(sess.run(w2))
#
#
#     # 设定训练的轮数
#     STEPS = 5000
#     for i in range(STEPS):
#         # 每次选取batch_size个样本进行训练
#         start = (i * batch_size) % dataset_size
#         end   = min(start + batch_size, dataset_size)
#
#         #通过选取的样本训练神经网络并更新参数
#         sess.run(train_step,
#                  feed_dict={x:X[start:end], y_:Y[start:end]})
#
#         if i % 1000 == 0:
#             # 每隔一段时间计算在所有数据上的交叉熵并输出
#             total_cross_entropy = sess.run(
#                 cross_entropy, feed_dict={x:X, y_:Y}
#             )
#             print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))
#
#
#     print(sess.run(w1))
#     print(sess.run(w2))








# w1 = tf.Variable(tf.random_normal([2,3], stddev=1))
# w2 = tf.Variable(tf.random_normal([3,1], stddev=1))
#
# x = tf.placeholder(tf.float32, shape=(3,2), name='input')
# a = tf.matmul(x, w1)
# y = tf.matmul(a, w2)
#
# y_ = tf.constant([1.2, 3.6,7.2])
#
# # 定义损失函数来刻画预测值与真实值的差距
# cross_entropy = -tf.reduce_mean(
#     y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
# )
#
# # 定义学习率
# learning_rate = 0.001
# # 定义反向传播算法来优化神经网络中的参数
# train_step =\
#     tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)
#
#
# sess = tf.Session()
# init_op = tf.global_variables_initializer()
# sess.run(init_op)
#
# print(sess.run(y, feed_dict={x:[[1,3], [2,3], [5,6.0]]}))


