import os
import cv2
import numpy
import Read_API
import tensorflow as tf
import pandas as pd

train_path = "D:/data1/data3/datas_face_train_feature"
test_path = "D:/data1/data3/datas_face_test_feature"
write_path = "D:/data1/data3/writer"
model_path = "D:/data1/data3/model"
TRAIN_CSV = ""
TEST_CSV = ""


def load_data(input_path):
    file_list = os.listdir(input_path)
    images = []
    for i, d in enumerate(file_list):
        try:
            img = cv2.imread(os.path.join(input_path, d))
            img = numpy.array(img, dtype=numpy.float32)
            cv2.normalize(img, img)
            images.append(img)
        except:
            print("Can not open%s" % os.path.join(input_path, d))
    return images


class CNN1(object):
    @staticmethod
    def lrelu(x, leak=0.2):
        return tf.maximum(x, x * leak)

    @staticmethod
    def add_convolution_layer(image, input_channel, filter_size, stddev, output_channel, strides, name,
                              active_function=None):
        with tf.name_scope(name):
            weight = tf.get_variable(name + 'W', filter_size + [input_channel, output_channel],
                                     initializer=tf.truncated_normal_initializer(stddev=stddev))
            # tf.summary.histogram(name + "\Filter", weight)
            basis = tf.get_variable(name + "B", output_channel, initializer=tf.constant_initializer(0.0))
            # tf.summary.histogram(name + "\B", basis)
            output = tf.nn.conv2d(image, weight, strides=strides, padding='SAME')
            if active_function:
                ac_output = active_function(tf.nn.bias_add(output, basis))
            else:
                ac_output = output
            # tf.summary.histogram(name + "\output", ac_output)
            return ac_output

    @staticmethod
    def add_pooling_layer(image, ksize, strides, name):
        with tf.name_scope(name):
            pooling = tf.nn.max_pool(image, ksize, strides, padding='SAME')
            return pooling

    @staticmethod
    def add_full_connection_layer(input_tensor, in_dim, out_dim, stddev, name, active_function=None):
        with tf.name_scope(name):
            weight = tf.get_variable(name + "W", [in_dim, out_dim],
                                     initializer=tf.truncated_normal_initializer(stddev=stddev))
            # tf.summary.histogram(name + "\W", weight)
            basis = tf.get_variable(name + "B", [out_dim], initializer=tf.constant_initializer(0.0))
            # tf.summary.histogram(name + "\B", basis)
            output = tf.add(tf.matmul(input_tensor, weight), basis)
            # tf.summary.histogram(name + "\out", output)
            if active_function:
                return active_function(output)
            else:
                return output

    @staticmethod
    def add_normalization_layer(input_tensor, name):
        with tf.name_scope(name):
            # tf.summary.histogram("input", input_tensor)
            output = tf.nn.l2_normalize(input_tensor, dim=1)
            # tf.summary.histogram("output", output)
        return output

    def __init__(self, input_height, input_width, input_channel, label_size, model_name):
        self.input_height = input_height
        self.input_width = input_width
        self.input_channel = input_channel
        self.label_size = label_size
        self.sess = tf.Session()
        self.tag_size = len(os.listdir(train_path))
        self.model_name = model_name

    def __del__(self):
        print("Info: model {} was deleted".format(self.model_name))

    def load_data(self, train_path, test_path, layer_path=None):
        self.data_pt, self.data_lt = self.load_test_data(test_path, sample_size=100, layer_path=layer_path)
        self.data_p, self.data_l = self.load_train_data(train_path, layer_path=layer_path)
        print("Info: model {} data load was completed".format(self.model_name))

    def build_model(self, reuse_flag):

        self.image_size = [self.input_height, self.input_width, self.input_channel]
        self.real_label = tf.placeholder(tf.float32, shape=[None, self.label_size], name='Real_Label')
        self.image_input = tf.placeholder(tf.float32, shape=[None] + self.image_size, name='Image')
        with tf.variable_scope(self.model_name, reuse=reuse_flag):
            # neural structure
            channel_size_list = [self.input_channel, 20, 40, 60, 80]
            # size[32，32，3]
            c1 = self.add_convolution_layer(self.image_input, input_channel=channel_size_list[0],
                                            output_channel=channel_size_list[1], filter_size=[3, 3], stddev=0.2,
                                            strides=[1, 1, 1, 1],
                                            active_function=self.lrelu, name="con_l1")
            p1 = self.add_pooling_layer(c1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name="pool_l1")
            # size [16,16,20]

            c2 = self.add_convolution_layer(p1, input_channel=channel_size_list[1],
                                            output_channel=channel_size_list[2],
                                            filter_size=[3, 3], stddev=0.02,
                                            strides=[1, 1, 1, 1],
                                            active_function=self.lrelu, name="con_l2", )
            p2 = self.add_pooling_layer(c2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name="pool_l2")
            # size [8,8,40]

            c3 = self.add_convolution_layer(p2, input_channel=channel_size_list[2], filter_size=[3, 3], stddev=0.02,
                                            output_channel=channel_size_list[3], strides=[1, 1, 1, 1],
                                            active_function=self.lrelu, name="con_l3")
            p3 = self.add_pooling_layer(c3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name="pool_l3")
            # size [4,4 60]

            c4 = self.add_convolution_layer(p3, input_channel=channel_size_list[3], filter_size=[3, 3], stddev=0.02,
                                            output_channel=channel_size_list[4], strides=[1, 1, 1, 1],
                                            name="con_l4",
                                            active_function=self.lrelu)

            p4 = self.add_pooling_layer(c4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name="pool_l4")
            # size [2,2,80]

            # full connection layer
            p3_flat = tf.reshape(p3, [-1, 4 * 4 * 60])
            p4_flat = tf.reshape(p4, [-1, 2 * 2 * 80])

            f1 = tf.concat([p3_flat, p4_flat], 1)
            nf1 = self.add_normalization_layer(f1, name="normalization")

            self.feature = self.add_full_connection_layer(nf1, 4 * 4 * 60 + 2 * 2 * 80, out_dim=100, stddev=0.1,
                                                          name="feature", active_function=self.lrelu)

            self.predict_label = self.add_full_connection_layer(self.feature, in_dim=100, out_dim=59, stddev=0.1,
                                                                name="label", active_function=tf.nn.softmax)

            self.cross_entropy = -tf.reduce_mean(
                tf.reduce_sum(self.real_label * tf.log(self.predict_label), reduction_indices=[1]))
            # tf.summary.scalar("loss", self.cross_entropy)
            self.train_step = tf.train.AdamOptimizer(0.01, name="Train").minimize(self.cross_entropy)

        self.correct_prediction = tf.equal(tf.arg_max(self.real_label, 1), tf.arg_max(self.predict_label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, dtype="float"))
        # tf.summary.scalar("AC", self.accuracy)
        self.merged = tf.summary.merge_all()
        if not os.path.exists(write_path):
            os.mkdir(write_path)
        if not os.path.exists(os.path.join(write_path, self.model_name)):
            os.mkdir(os.path.join(write_path, self.model_name))
        self.writer = tf.summary.FileWriter(os.path.join(write_path, self.model_name), self.sess.graph)
        self.saver = tf.train.Saver()

    def initial_para(self):
        init = tf.initialize_all_variables()
        self.sess.run(init)

    def save_para(self):
        if not os.path.exists(os.path.join(model_path, self.model_name)):
            os.mkdir(os.path.join(model_path, self.model_name))
        save_path = self.saver.save(self.sess, os.path.join(model_path, self.model_name, self.model_name))
        print("Info: save model at {}".format(save_path))

    def restore_para(self):
        self.saver.restore(self.sess, os.path.join(model_path, self.model_name, self.model_name))
        print("Info: restore model from {}".format(os.path.join(model_path, self.model_name, self.model_name)))

    def get_variable(self, variable, feed_dict):
        return self.sess.run(variable, feed_dict=feed_dict)

    def write_variable(self, index, feed_dict):
        result = self.sess.run(self.merged, feed_dict=feed_dict)
        self.writer.add_summary(result, index)

    def get_feature(self, image):
        if image is None:
            return numpy.zeros(100)
        return self.sess.run(self.feature,
                             feed_dict={self.image_input: [image], self.real_label: [numpy.zeros(self.label_size)]})[0]

    def load_test_data(self, path, layer_path=None, sample_size=None):
        data_t = []
        if layer_path is None:
            for i, people_name in enumerate(os.listdir(path)):
                tag = numpy.zeros([self.tag_size], dtype=numpy.float32)
                tag[i] = 1.0
                data_t += Read_API.load_all_pictures_in_file(os.path.join(path, people_name), dst_size=32, tag=tag)
        else:
            for i, people_name in enumerate(os.listdir(path)):
                tag = numpy.zeros([self.tag_size], dtype=numpy.float32)
                tag[i] = 1.0
                data_t += Read_API.load_all_pictures_in_file(os.path.join(path, people_name, layer_path), dst_size=32,
                                                             tag=tag)
        numpy.random.shuffle(data_t)
        data_t = data_t[:sample_size]
        data_pt = [p[0] for p in data_t]
        data_lt = [l[1] for l in data_t]
        print("Info: test data shape {}".format(numpy.array(data_pt).shape))
        return data_pt, data_lt

    def load_train_data(self, path, layer_path=None):
        data_t = []
        if layer_path is None:
            for i, people_name in enumerate(os.listdir(path)):
                tag = numpy.zeros([self.tag_size], dtype=numpy.float32)
                tag[i] = 1.0
                data_t += Read_API.load_all_pictures_in_file(os.path.join(path, people_name), dst_size=32, tag=tag)
        else:
            for i, people_name in enumerate(os.listdir(path)):
                tag = numpy.zeros([self.tag_size], dtype=numpy.float32)
                tag[i] = 1.0
                data_t += Read_API.load_all_pictures_in_file(os.path.join(path, people_name, layer_path), dst_size=32,
                                                             tag=tag)
        numpy.random.shuffle(data_t)
        data_t = data_t
        data_p = [p[0] for p in data_t]
        data_l = [l[1] for l in data_t]
        print("Info: train data shape {}".format(numpy.array(data_p).shape))
        return data_p, data_l

    def loadTrainDataFromDisk(self, batch_size):
        pd.read_csv(TRAIN_CSV)

    def trainInMemory(self, step, batchsize):
        print("Info: model {} train process start".format(self.model_name))
        trainsamplesize = len(self.data_p)
        for i in range(step):
            train_start = (batchsize * i) % trainsamplesize
            train_end = (batchsize * (i + 1)) % trainsamplesize
            if train_end < train_start:
                continue
            train_feed = {self.image_input: self.data_p[train_start:train_end],
                          self.real_label: self.data_l[train_start:train_end]}
            test_feed = {self.image_input: self.data_pt, self.real_label: self.data_lt}
            self.sess.run(self.train_step, feed_dict=train_feed)
            # self.write_variable(index=i, feed_dict=test_feed)

            if i % 100 == 0:
                print("\t Step {}".format(i))
                print("\t CE {}".format(self.get_variable(self.cross_entropy, feed_dict=test_feed)))
                print("\t AC {}".format(self.get_variable(self.accuracy, feed_dict=test_feed)))
        return self.get_variable(self.accuracy, feed_dict=test_feed)

    def trainInDisk(self, step, batch_size):
        print("Info: model {} train process start(in disk)".format(self.model_name))
        for i in range(step):
            train_start = (batchsize * i) % trainsamplesize
            train_end = (batchsize * (i + 1)) % trainsamplesize
            if train_end < train_start:
                continue
            train_feed = {self.image_input: self.data_p[train_start:train_end],
                          self.real_label: self.data_l[train_start:train_end]}
            test_feed = {self.image_input: self.data_pt, self.real_label: self.data_lt}
            self.sess.run(self.train_step, feed_dict=train_feed)
            # self.write_variable(index=i, feed_dict=test_feed)

            if i % 100 == 0:
                print("\t Step {}".format(i))
                print("\t CE {}".format(self.get_variable(self.cross_entropy, feed_dict=test_feed)))
                print("\t AC {}".format(self.get_variable(self.accuracy, feed_dict=test_feed)))
        return self.get_variable(self.accuracy, feed_dict=test_feed)


if __name__ == "__main__":
    namelist = os.listdir(train_path)
    model_list = os.listdir(os.path.join(train_path, os.listdir(train_path)[0]))
    print("Info: model_amount {}".format(len(model_list)))
    for i, model_name in enumerate(model_list):
        if i < 74:
            continue
        C1 = CNN1(input_height=32, input_width=32, label_size=59, input_channel=1, model_name=model_name)
        C1.build_model(False)
        C1.initial_para()
        C1.load_data(train_path, test_path, layer_path=model_name)
        AC = C1.train(batchsize=50, step=5000)
        output = open("data3/result.txt", 'a')
        output.write("model {}: ac {} \n".format(model_name, AC))
        output.close()
        C1.save_para()
        del C1
