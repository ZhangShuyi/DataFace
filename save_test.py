import tensorflow as tf
import numpy as np


# 保存神经网络参数
def save_para():
    # 定义权重参数
    W = tf.Variable([[1, 2, 3], [4, 5, 6]], dtype=tf.float32, name='weights')
    # 定义偏置参数
    b = tf.Variable([[1, 2, 3]], dtype=tf.float32, name='biases')
    # 参数初始化
    init = tf.global_variables_initializer()
    # 定义保存参数的saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        # 保存session中的数据
        save_path = saver.save(sess, "D:\\my_net\\save_net.ckpt")
        # 输出保存路径
        print('Save to path: ', save_path)


# 恢复神经网络参数
def restore_para():
    # 定义权重参数
    W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name='weights')
    # 定义偏置参数
    b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name='biases')
    # 定义提取参数的saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # 加载文件中的参数数据，会根据name加载数据并保存到变量W和b中
        saver.restore(sess, "D:\\my_net\\save_net.ckpt")
        # 输出保存路径
        print('Weights: ', sess.run(W))
        print('biases:  ', sess.run(b))


#save_para()
restore_para()
