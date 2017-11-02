import os
import Read_API
import tensorflow as tf
import numpy

train_path = 'D:\\data1\\datas_face4'
test_path = 'D:\\data1\\datas_face4_test'

train_path_positive = "D:\\data1\\datas_face4\\RachaelRay"
test_path_positive = "D:\\data1\\datas_face4_test\\RachaelRay"


def filter_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def convolution_2d(x, f):
    return tf.nn.conv2d(x, filter=f, strides=[1, 1, 1, 1], padding='SAME')


def pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def add_con_layer(inputs, filter_size, n_layer, activation_function=None):
    layer_name = "con_layer%s" % n_layer
    with tf.name_scope(layer_name):
        filter_l = filter_variable(filter_size)
        b_l = bias_variable([filter_size[3]])
        tf.summary.histogram("bias", b_l)
        h_l = convolution_2d(inputs, filter_l)
        tf.summary.histogram("h_l", h_l)
        if activation_function is None:
            output = h_l
        else:
            output = activation_function(h_l)
        tf.summary.histogram("outputs", output)
        return output


def add_pool_layer(inputs, ksize, stride, n_layer):
    layer_name = "pool_layer%s" % n_layer
    with tf.name_scope(layer_name):
        output = tf.nn.max_pool(inputs, ksize, stride, padding='SAME')
        # tf.summary.image(layer_name+"/outputs")
    return output


def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
    layer_name = "layer%s" % n_layer
    with tf.name_scope(layer_name):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]))
        tf.summary.histogram("weights", Weights)
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
        tf.summary.histogram("biases", biases)
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
        tf.summary.histogram("Wx_plus_b", Wx_plus_b)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram("outputs", outputs)
        return outputs


with tf.name_scope("data_in"):
    x_image_input_1 = tf.placeholder(tf.float32, shape=[None, 64, 64, 1])
    # x_image_input_2 = tf.placeholder(tf.float32, shape=[None, 64, 64, 1])
    y_label = tf.placeholder(tf.float32, shape=[None, 2])

# convolution_1
h_l1 = add_con_layer(x_image_input_1, filter_size=[5, 5, 1, 32], n_layer="1", activation_function=tf.nn.relu)
hp_l1 = add_pool_layer(h_l1, n_layer="1", ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1])
# convolution_2
h_l2 = add_con_layer(hp_l1, filter_size=[5, 5, 32, 64], n_layer="2", activation_function=tf.nn.relu)
hp_l2 = add_pool_layer(h_l2, n_layer="2", ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1])

hp_l2_flat = tf.reshape(hp_l2, [-1, 16 * 16 * 64])

h_fc1 = add_layer(hp_l2_flat, in_size=16 * 16 * 64, out_size=1024, n_layer="1", activation_function=tf.nn.softmax)
h_fc2 = add_layer(h_fc1, in_size=1024, out_size=512, n_layer="2", activation_function=tf.nn.softmax)
h_fc3 = add_layer(h_fc2, in_size=512, out_size=256, n_layer="3", activation_function=tf.nn.softmax)
h_fc4 = add_layer(h_fc3, in_size=256, out_size=256, n_layer="4", activation_function=tf.nn.softmax)
label_out = add_layer(h_fc4, in_size=256, out_size=2, n_layer="5", activation_function=tf.nn.softmax)

with tf.name_scope("cross_entropy"):
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y_label * tf.log(label_out), reduction_indices=[1]))
    tf.summary.scalar("cross_entropy", cross_entropy)
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(0.0003).minimize(cross_entropy)

with tf.name_scope("ac"):
    correct_prediction = tf.equal(tf.arg_max(label_out, 1), tf.arg_max(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("ac", accuracy)

init = tf.initialize_all_variables()
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("CNN4/", sess.graph)
sess.run(init)

if __name__ == "__main__":
    data_train_P = Read_API.load_data(train_path_positive, [1, 0])
    data_train_N = []
    for dectories in os.listdir(train_path):
        if os.path.join(train_path, dectories) == train_path_positive:
            continue
        else:
            data_train_N += Read_API.load_data(os.path.join(train_path, dectories), [0, 1])
    print("train P%d train N%d" % (len(data_train_P), len(data_train_N)))
    print("Loading train data finished!")

    data_test_P = Read_API.load_data(test_path_positive, [1, 0])
    data_test_N = []
    for dectories in os.listdir(test_path):
        if os.path.join(test_path, dectories) == test_path_positive:
            continue
        else:
            data_test_N += Read_API.load_data(os.path.join(test_path, dectories), [0, 1])
    print("test P%d test N%d" % (len(data_test_P), len(data_test_N)))
    print("Loading test data finished!")

    numpy.random.shuffle(data_test_N)
    numpy.random.shuffle(data_test_P)
    dataP_t = numpy.array([p[0] for p in data_test_P[:35]] + [p[0] for p in data_test_N[:35]])
    dataL_t = numpy.array([l[1] for l in data_test_P[:35]] + [l[1] for l in data_test_N[:35]])

    step = 50
    for i in range(500):
        numpy.random.shuffle(data_train_N)
        numpy.random.shuffle(data_train_P)
        dataP = numpy.array([p[0] for p in data_train_P[:step]] + [p[0] for p in data_train_N[:step]])
        dataL = numpy.array([l[1] for l in data_train_P[:step]] + [l[1] for l in data_train_N[:step]])
        sess.run(train_step, feed_dict={x_image_input_1: dataP, y_label: dataL})
        print("ce %f" % sess.run(cross_entropy, feed_dict={x_image_input_1: dataP, y_label: dataL}))
        result = sess.run(merged, feed_dict={x_image_input_1: dataP_t[:70], y_label: dataL_t[:70]})
        writer.add_summary(result, i)
