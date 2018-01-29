# -*- coding: utf-8 -*-
import tensorflow as tf

#设置提示信息的级数（只显示 warning 和 Error）
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# 定义神经网络结构相关的参数
INPUT_NODE  = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight_variable(shape, regularizer):
    weights = tf.get_variable(
        'weights', shape,
        initializer=tf.random_normal_initializer(stddev=0.1))

    if regularizer != None:
        tf.add_to_collection('losses', regularizer(weights))
    return weights



def inference(input_tensor, regularizer):
    # 声明第一层神经网络的变量并完成前向传输过程
    with tf.variable_scope('layer1'):
        weights = get_weight_variable(
            [INPUT_NODE, LAYER1_NODE], regularizer)

        biases = tf.get_variable('biases', [LAYER1_NODE],
                                 initializer=tf.constant_initializer(0.0))

        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    # 类似的声明第二层神经网络的变量并完成前向传播过程
    with tf.variable_scope('layer2'):
        weights = get_weight_variable(
            [LAYER1_NODE, OUTPUT_NODE], regularizer)

        biases  = tf.get_variable(
            'biases', [OUTPUT_NODE],
            initializer=tf.constant_initializer(0.0))

        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)

    return layer2