import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
#
# # 定义常量
# a = tf.constant([1.0, 2.0], name='a')
# b = tf.constant([2.0, 3.0], name='b')
# result = a + b
#
# # 可以
# # print(a.graph is tf.get_default_graph())
#
# sess = tf.Session()
# print(sess.run(result))


g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable(
        'v', initializer=tf.zeros_initializer, shape=[1])

g2 = tf.Graph()
with g2.as_default():
    v = tf.get_variable(
        'v', initializer=tf.ones_initializer, shape=[1])

config = tf.ConfigProto(allow_soft_placement=True)
#最多占gpu资源的70%
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
#开始不会给tensorflow全部gpu资源 而是按需增加
config.gpu_options.allow_growth = True

# 在计算图一中读取变量'v'的值
with tf.Session(graph=g1, config=config) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable('v')))

# 在计算图二中读取变量'v'的值
with tf.Session(graph=g2, config=config) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable('v')))
