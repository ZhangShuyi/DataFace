import os
import tensorflow as tf
import cv2
import numpy

train_path = 'D:\\data1\\datas_face4'
test_path = 'D:\\data1\\datas_face4_test'


def load_data(data_path):
    data = []
    dirnamelist = os.listdir(data_path)
    print("Person Number %d" % len(dirnamelist))
    for lab, dirname in enumerate(dirnamelist):
        filenamelist = os.listdir(os.path.join(data_path, dirname))
        for index, filename in enumerate(filenamelist):
            data_line = []
            try:
                img = cv2.imread(os.path.join(data_path, dirname, filename))
                img = numpy.array(img[:, :, :1], dtype=numpy.float32)
                cv2.normalize(img, img)
                data_line.append(img)
                lab_vector = numpy.array(numpy.zeros(len(dirnamelist) + 1))
                lab_vector[lab] = 1
                data_line.append(lab_vector)
            except FileNotFoundError:
                print("File Not Found Error")
            data.append(data_line)
    print("Randomize data ")
    numpy.random.shuffle(data)
    return data


def filter_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.001, shape=shape)
    return tf.Variable(initial)


def convolution_2d(x, f):
    return tf.nn.conv2d(x, filter=f, strides=[1, 1, 1, 1], padding='SAME')


def pooling_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# define network parameter
image_input_size = 64
image_input_channel = 1

# define network structure
x_image_input = tf.placeholder(tf.float32, shape=[None, 64, 64, 1])
y_label = tf.placeholder(tf.float32, shape=[None, 60])

filter_l1 = filter_variable([5, 5, 1, 32])
b_l1 = bias_variable([32])

h_l1 = tf.nn.relu(convolution_2d(x_image_input, filter_l1) + b_l1)
hp_l1 = pooling_2x2(h_l1)

filter_l2 = filter_variable([5, 5, 32, 64])
b_l2 = bias_variable([64])

h_l2 = tf.nn.relu(convolution_2d(hp_l1, filter_l2) + b_l2)
hp_l2 = pooling_2x2(h_l2)
hp_l2_flat = tf.reshape(hp_l2, [-1, 16 * 16 * 64])

w_fc1 = filter_variable([16 * 16 * 64, 1024])
b_fc1 = bias_variable([1024])
h_fc1 = tf.nn.relu(tf.matmul(hp_l2_flat, w_fc1) + b_fc1)

w_fc2 = filter_variable([1024, 1024])
b_fc2 = bias_variable([1024])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2)

w_fc5 = filter_variable([1024, 60])
b_fc5 = bias_variable([60])
label_out = tf.nn.softmax(tf.matmul(h_fc2, w_fc5) + b_fc5)

# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=label_out, labels=y_label))
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y_label * tf.log(label_out), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(0.003).minimize(cross_entropy)

correct_prediction = tf.equal(tf.arg_max(label_out, 1), tf.arg_max(y_label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

data = load_data(train_path)
print("Loading train data finished!")

data_t = load_data(test_path)
print("Loading test data finished!")
dataP_t = [p[0] for p in data_t]
dataL_t = [l[1] for l in data_t]

step = 100
for i in range(1000):
    # print(sess.run(x_image_input, feed_dict={x_image_input: dataP[0:1], y_label: dataL[0:1]}))
    # print(sess.run(y_label, feed_dict={x_image_input: dataP[0:1], y_label: dataL[0:1]}))
    # print(sess.run(label_out, feed_dict={x_image_input: dataP[0:1], y_label: dataL[0:1]}))
    numpy.random.shuffle(data)
    dataP = [p[0] for p in data[:step]]
    dataL = [l[1] for l in data[:step]]
    train_step.run(
        feed_dict={x_image_input: dataP, y_label: dataL})
    print(
        "cross_entropy %f" % sess.run(cross_entropy, feed_dict={x_image_input: dataP, y_label: dataL}))
    print("ac: %f" % sess.run(accuracy, feed_dict={x_image_input: dataP_t[:200], y_label: dataL_t[:200]}))


'''print(sess.run(hp_l1, feed_dict={x_image_input: dataP[1:2], y_label: dataL[1:2]}))
print("\n")
print(sess.run(hp_l2, feed_dict={x_image_input: dataP[1:2], y_label: dataL[1:2]}))
print("\n")
print(sess.run(h_fc1, feed_dict={x_image_input: dataP[1:2], y_label: dataL[1:2]}))
print("\n")
print(sess.run(h_fc2, feed_dict={x_image_input: dataP[1:2], y_label: dataL[1:2]}))
print(sess.run(label_out, feed_dict={x_image_input: dataP[1:1 + 1], y_label: dataL[1:1 + 1]}))
'''
