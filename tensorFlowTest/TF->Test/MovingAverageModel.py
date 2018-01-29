#设置提示信息的级数（只显示 warning 和 Error）
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

v1 = tf.Variable(0, dtype=tf.float32)

step = tf.Variable(0, trainable=False)

ema = tf.train.ExponentialMovingAverage(0.99, step)

maintain_average_op = ema.apply([v1])

with tf.Session() as sess:
    # 初始化所有变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 通过ema.average(v1)获取滑动平均之后变量的取值。在初始化之后变量v1的值和v1
    # 的滑动平均都为0
    print(sess.run([v1, ema.average(v1)]))

    # 更新变量v1的值到5
    sess.run(tf.assign(v1, 5))
    # 更新v1的滑动平均值。衰减率为min( 0.99, (1+step)/(10+step)=0.1 ) = 0.1
    # 所以v1的滑动平均会被更新为0.1*0 + 0.9*5 = 4.5
    sess.run(maintain_average_op)
    print(sess.run([v1, ema.average(v1)]))

    # 更新step的值为10000
    sess.run(tf.assign(step, 10000))
    # 跟新v1的值为10
    sess.run(tf.assign(v1, 10))

    sess.run(maintain_average_op)
    print(sess.run([v1, ema.average(v1)]))
    # 输出[10.0, 4.5549998]

    # 再次更新滑动平均值， 得到的新滑动平均值为0.99*4.555+0.01+10=4.60945
    sess.run(maintain_average_op)
    print(sess.run([v1, ema.average(v1)]))
