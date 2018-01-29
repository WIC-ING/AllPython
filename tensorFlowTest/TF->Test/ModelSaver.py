#设置提示信息的级数（只显示 warning 和 Error）
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# export TF_CPP_MIN_LOG_LEVEL=2

# 导入tensorflow与numpy
import tensorflow as tf

# saver = tf.train.import_meta_graph(
#     'model/model.ckpt.meta')
#
# with tf.Session() as sess:
#     saver.restore(sess, 'model/model.ckpt')
#     print(sess.run(tf.get_default_graph().get_tensor_by_name('add:0')))




v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='other-v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='other-v2')
result = v1 + v2

init_op = tf.global_variables_initializer()
# saver = tf.train.Saver()

# 重命名变量：将原来模型中的变量映射到当前的变量中
saver = tf.train.Saver({'v1':v1, 'v2':v2})

with tf.Session() as sess:
    # 保存模型
    # sess.run(init_op)
    # saver.save(sess, 'model/model.ckpt')

    # 加载模型
    saver.restore(sess, 'model/model.ckpt')
    print(sess.run(result))