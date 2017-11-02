import os
import Read_API
import tensorflow as tf
import numpy
import random

train_path = 'D:\\data1\\datas_face4'
test_path = 'D:\\data1\\datas_face4_test'

train_path_positive = "D:\\data1\\datas_face4\\RachaelRay"
test_path_positive = "D:\\data1\\datas_face4_test\\RachaelRay"


def load_verification_data(path):
    data = []
    for i, file in enumerate(os.listdir(path)):
        data.append(Read_API.load_data(os.path.join(path, file), tag=[i]))
    print("class :%d" % len(data))
    return data


def extract_positive_sample(data):
    main_index = numpy.random.randint(0, len(data))
    main_length = len(data[main_index])
    index = random.sample(range(main_length), 2)
    return [data[main_index][index[0]][0], data[main_index][index[1]][0]]


def extract_negative_sample(data):
    main_index = random.sample(range(len(data)), 2)
    index_0 = numpy.random.randint(0, len(data[main_index[0]]))
    index_1 = numpy.random.randint(0, len(data[main_index[1]]))
    return [data[main_index[0]][index_0][0], data[main_index[1]][index_1][0]]


def extract_by_step(data, step):
    sample = []
    for i in range(step):
        sample.append([extract_positive_sample(data), [1, 0]])
        sample.append([extract_negative_sample(data), [0, 1]])
    numpy.random.shuffle(sample)
    return numpy.array(sample)


def filter_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, mean=0.0)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=1.0, dtype=tf.float32)
    return tf.Variable(initial)


def convolution_2d(x, f):
    return tf.nn.conv2d(x, filter=f, strides=[1, 1, 1, 1], padding='SAME')


def pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def add_con_layer(inputs, kernel_filter, n_layer, activation_function=None):
    layer_name = "con_layer%s" % n_layer
    with tf.name_scope(layer_name):
        b_l = bias_variable([int(kernel_filter.shape[3])])
        tf.summary.histogram("bias", b_l)
        h_l = convolution_2d(inputs, kernel_filter) + b_l
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
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.01)
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
    x_image_input_2 = tf.placeholder(tf.float32, shape=[None, 64, 64, 1])
    y_label = tf.placeholder(tf.float32, shape=[None, 2])

# convolution_1
kernel_filter1_layer1 = filter_variable([5, 5, 1, 32])
h1_l1 = add_con_layer(x_image_input_1, kernel_filter1_layer1, n_layer="1.1", activation_function=tf.nn.relu)
hp1_l1 = add_pool_layer(h1_l1, n_layer="1.1", ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1])
# convolution_2
kernel_filter1_layer2 = filter_variable([5, 5, 32, 64])
h1_l2 = add_con_layer(hp1_l1, kernel_filter1_layer2, n_layer="1.2", activation_function=tf.nn.softmax)
hp1_l2 = add_pool_layer(h1_l2, n_layer="1.2", ksize=[1, 4, 4, 1], stride=[1, 4, 4, 1])

# convolution_1
kernel_filter2_layer1 = filter_variable([5, 5, 1, 32])
h2_l1 = add_con_layer(x_image_input_2, kernel_filter2_layer1, n_layer="2.1", activation_function=tf.nn.softmax)
hp2_l1 = add_pool_layer(h2_l1, n_layer="2.1", ksize=[1, 2, 2, 1], stride=[1, 2, 2, 1])
# convolution_2
kernel_filter2_layer2 = filter_variable([5, 5, 32, 64])
tf.summary.histogram("filter2,2", kernel_filter2_layer2)
h2_l2 = add_con_layer(hp2_l1, kernel_filter2_layer2, n_layer="2.2", activation_function=tf.nn.softmax)
hp2_l2 = add_pool_layer(h2_l2, n_layer="2.2", ksize=[1, 4, 4, 1], stride=[1, 4, 4, 1])

hp1_l2_flat = tf.reshape(hp1_l2, [-1, 8 * 8 * 64])
hp2_l2_flat = tf.reshape(hp2_l2, [-1, 8 * 8 * 64])
hp_l2_flat = tf.concat([hp1_l2_flat, hp2_l2_flat], 1)

h_fc1 = add_layer(hp_l2_flat, in_size=2 * 8 * 8 * 64, out_size=512, n_layer="1", activation_function=tf.nn.softmax)
label_out = add_layer(h_fc1, in_size=512, out_size=2, n_layer="5", activation_function=tf.nn.softmax)

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
    data_train = load_verification_data(train_path)
    data_test = load_verification_data(test_path)
    step = 5
    test_step = 50
    test_sample = extract_by_step(data_test, test_step)
    for i in range(50000):
        print("step %d" % i)
        train_sample = extract_by_step(data_train, step)
        sess.run(train_step, feed_dict={x_image_input_1: [p[0][0] for p in train_sample],
                                        x_image_input_2: [p[0][1] for p in train_sample],
                                        y_label: [l[1] for l in train_sample]})

        print(sess.run(kernel_filter1_layer1, feed_dict={x_image_input_1: [p[0][0] for p in train_sample],
                                                         x_image_input_2: [p[0][1] for p in train_sample],
                                                         y_label: [l[1] for l in train_sample]}))
        result = sess.run(merged, feed_dict={x_image_input_1: [p[0][0] for p in test_sample],
                                             x_image_input_2: [p[0][1] for p in test_sample],
                                             y_label: [l[1] for l in test_sample]})
        writer.add_summary(result, i)
