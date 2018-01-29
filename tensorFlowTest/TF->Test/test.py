#设置提示信息的级数（只显示 warning 和 Error）
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# export TF_CPP_MIN_LOG_LEVEL=2

# 导入tensorflow与numpy
import tensorflow as tf
import numpy as np


with tf.variable_scope('first'):
    x = tf.get_variable('a', [1])

with tf.variable_scope('seconde'):
    x = tf.get_variable('a', [1])

print(x)



# # 加载模型，并重命名变量
# # v = tf.Variable(0, dtype=tf.float32, name='v')
# #
# # saver = tf.train.Saver({'v/ExponentialMovingAverage':v})
# #
# # with tf.Session() as sess:
# #     saver.restore(sess, 'model/model.ckpt')
# #     print(sess.run(v))
#
#
#
# # 构造并保存模型
# v = tf.Variable(0, dtype=tf.float32, name='v')
# #
# # for variable in tf.global_variables():
# #     print(variable.name)
#
# ema = tf.train.ExponentialMovingAverage(0.99)
# # maintain_averages_op = ema.apply(tf.global_variables())
#
# print(ema.variables_to_restore())
#
# saver = tf.train.Saver(ema.variables_to_restore())
# with tf.Session() as sess:
#     saver.restore(sess, 'model/model.ckpt')
#     print(sess.run(v))


# for variable in tf.global_variables():
#     print(variable.name)
#
# saver = tf.train.Saver()
# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#
#     sess.run(tf.assign(v, 10))
#     sess.run(maintain_averages_op)
#
#     saver.save(sess, 'model/model.ckpt')
#     print(sess.run([v, ema.average(v)]))










# with tf.variable_scope('root'):
#     print(tf.get_variable_scope().reuse)
#
#     with tf.variable_scope('foo', reuse=True):
#
#         print(tf.get_variable_scope().reuse)
#         with tf.variable_scope('bar'):
#             print(tf.get_variable_scope().reuse)
#
#     print(tf.get_variable_scope().reuse)


# v1 = tf.get_variable('v', [1])
#
# print(v1.name)
#
#
# with tf.variable_scope('foo'):
#     v2 = tf.get_variable('v', [1])
#     print(v2.name)
#
#
# with tf.variable_scope('foo'):
#     with tf.variable_scope('bar'):
#         v3 = tf.get_variable('v', [1])
#         print(v3.name)
#
#
#     v4 = tf.get_variable('v1', [1])
#     print(v4.name)
#
# # 创建一个名称为空的命名空间，并设置reuse=True
# with tf.variable_scope('', reuse=True):
#     v5 = tf.get_variable('foo/bar/v', [1])
#
#     print('v5==v3: ', v5 == v3)
#
#     v6 = tf.get_variable('foo/v1', [1])
#     print('v6==v4: ', v6==v4)

# y2 = tf.convert_to_tensor([[0, 0, 1, 0]], dtype=tf.int64)
# y_2 = tf.convert_to_tensor([[-2.6, -1.7, 3.2, 0.1]], dtype=tf.float32)
# c2 = tf.nn.sparse_softmax_cross_entropy_with_logits(_sentinel=None, logits=y_2, labels=tf.argmax(y2, 1), name=None)
#
# sess = tf.Session()
# print(sess.run(c2))

# def get_weight(shape, lamb):
#     var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
#     tf.add_to_collection(
#         'losses', tf.contrib.layers.l2_regularizer(lamb)(var)
#     )
#
#     return var
#
# x  = tf.placeholder(tf.float32, shape=(None, 2))
# y_ = tf.placeholder(tf.float32, shape=(None, 1))
#
# # 定义了每一层网络中节点的个数
# layer_dimension = [2, 10, 10, 10, 1]
# # 神经网络的层数
# n_layers = len(layer_dimension)
#
# # 这个变量维护前向传播时传播最深层的节点，开始的时候就是输入层
# cur_layer = x
# # 当前层的节点个数
# in_dimension = layer_dimension[0]
#
# # 通过一个循环来生成5层全连接的神经网络结构
# for i in range(1, n_layers):
#     # layer_dimension[i] 为下一层的节点个数
#     out_dimension = layer_dimension[i]
#
#     # 生成当前层中权重的变量，并将这个变量的L2正则化损失加入计算图上的集合
#     weight = get_weight([in_dimension, out_dimension], 0.001)
#     bias   = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
#
#     # 使用ReLU激活函数
#     cur_layer = tf.nn.relu( tf.matmul(cur_layer, weight) + bias )
#
#     # 在进入下一层前将下一层的节点个数更新为当前层节点个数
#     in_dimension = layer_dimension[i]
#
#
# # 在定义神经网络前向传播的同时已经将所有的L2正则化损失加入了土山的集合
# # 这里只需要计算刻画模型在训练数据上表现的损失函数
# mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
#
# # 将均方误差损失函数加入损失集合
# tf.add_to_collection('losses', mse_loss)
#
# loss = tf.add_n(tf.get_collection('losses'))





# weights = tf.constant([1.0, -2.0])
# with tf.Session() as sess:
#     print(sess.run(tf.contrib.layers.l1_regularizer(0.5)(weights)))
#     print(sess.run(tf.contrib.layers.l2_regularizer(0.5)(weights)))