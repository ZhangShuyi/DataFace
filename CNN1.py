import os
import numpy
from InternalModule.Header import *
import tensorflow as tf
import pandas as pd

TR_PATH = "F:/Feature"
TS_PATH = ""
WR_PATH = ""
RE_PATH = "F:"
MD_PATH = "F:/Model"
TR_CSV = "F:/CSV"
PEOPLE_SIZE = 1042


class CNN1(object):
    @staticmethod
    def lrelu(x, leak=0.2):
        return tf.maximum(x, x * leak)

    @staticmethod
    def add_convolution_layer(image, input_channel, filter_size, stddev, output_channel, strides, name,
                              active_function=None, record=False):
        with tf.name_scope(name):
            weight = tf.get_variable(name + 'W', filter_size + [input_channel, output_channel],
                                     initializer=tf.truncated_normal_initializer(stddev=stddev))
            basis = tf.get_variable(name + "B", output_channel, initializer=tf.constant_initializer(0.0))
            export = tf.nn.conv2d(image, weight, strides=strides, padding='SAME')
            if active_function:
                ac_output = active_function(tf.nn.bias_add(export, basis))
            else:
                ac_output = export
            if record:
                tf.summary.histogram(name + "\Filter", weight)
                tf.summary.histogram(name + "\B", basis)
                tf.summary.histogram(name + "\export", ac_output)
            return ac_output

    @staticmethod
    def add_pooling_layer(image, ksize, strides, name, record=False):
        with tf.name_scope(name):
            pooling = tf.nn.max_pool(image, ksize, strides, padding='SAME')
            if record:
                tf.summary.histogram(name + "\pooling", pooling)
        return pooling

    @staticmethod
    def add_full_connection_layer(input_tensor, in_dim, out_dim, stddev, name, active_function=None, record=False):
        with tf.name_scope(name):
            weight = tf.get_variable(name + "W", [in_dim, out_dim],
                                     initializer=tf.truncated_normal_initializer(stddev=stddev))
            basis = tf.get_variable(name + "B", [out_dim], initializer=tf.constant_initializer(0.0))
            export = tf.add(tf.matmul(input_tensor, weight), basis)
            if record:
                tf.summary.histogram(name + "\W", weight)
                tf.summary.histogram(name + "\B", basis)
                tf.summary.histogram(name + "\out", export)
            if active_function:
                return active_function(export)
            else:
                return export

    @staticmethod
    def add_normalization_layer(input_tensor, name, record=False):
        with tf.name_scope(name):
            export = tf.nn.l2_normalize(input_tensor, dim=1)
            if record:
                tf.summary.histogram("input", input_tensor)
                tf.summary.histogram("output", export)
        return export

    def __init__(self, input_size, label_size, model_name):
        try:
            self.input_height = input_size[0]
            self.input_width = input_size[1]
            self.input_channel = input_size[2]
        except KeyError:
            ROOT_LOG.error("Model {}'s input size is invalid".format(model_name))
            return
        self.model_name = model_name
        self.label_size = label_size
        self.tag_size = len(os.listdir(TR_PATH))
        self.sess = tf.Session()
        ROOT_LOG.info("Model {} init success".format(self.model_name))

    def build_model(self, reuse_flag, record=False):
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

            self.predict_label = self.add_full_connection_layer(self.feature, in_dim=100, out_dim=self.label_size,
                                                                stddev=0.1,
                                                                name="label", active_function=tf.nn.softmax)

            self.loss_function = tf.reduce_sum(tf.square(tf.subtract(self.real_label, self.predict_label)))
            # self.loss_function = -tf.reduce_mean(
            #    tf.reduce_sum(self.real_label * tf.log(self.predict_label), reduction_indices=[1]))
            self.train_step = tf.train.AdamOptimizer(0.001, name="Train").minimize(self.loss_function)

        self.correct_prediction = tf.equal(tf.arg_max(self.real_label, 1), tf.arg_max(self.predict_label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, dtype="float"))
        if record:
            tf.summary.scalar("AC", self.accuracy)
            tf.summary.scalar("loss", self.loss_function)
            self.merged = tf.summary.merge_all()
            if not os.path.exists(WR_PATH):
                os.mkdir(WR_PATH)
            if not os.path.exists(os.path.join(WR_PATH, self.model_name)):
                os.mkdir(os.path.join(WR_PATH, self.model_name))
            self.writer = tf.summary.FileWriter(os.path.join(WR_PATH, self.model_name), self.sess.graph)

        self.saver = tf.train.Saver()

    def initial_para(self):
        init = tf.initialize_all_variables()
        self.sess.run(init)

    def save_para(self):
        if not os.path.exists(os.path.join(MD_PATH, self.model_name)):
            os.mkdir(os.path.join(MD_PATH, self.model_name))
        save_path = self.saver.save(self.sess, os.path.join(MD_PATH, self.model_name, self.model_name))
        print("Info: save model at {}".format(save_path))

    def restore_para(self):
        self.saver.restore(self.sess, os.path.join(MD_PATH, self.model_name, self.model_name))
        print("Info: restore model from {}".format(os.path.join(MD_PATH, self.model_name, self.model_name)))

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

    def loadDataInMemory(self, path, csv_path, number_limit=0):
        csv_data = pd.read_csv(csv_path).values
        numpy.random.shuffle(csv_data)
        tr_data = []
        tr_lab = []
        ts_data = []
        ts_lab = []
        train_num = 0
        test_num = 0
        for index, line in enumerate(csv_data):
            line = line[0]
            [number, people_name, pic_name, train_tag] = str(line).split('w')
            if number_limit != 0 and index > number_limit:
                break
            lab = numpy.zeros(self.label_size)
            lab[int(people_name) - 1] = 1.0
            img = ReadPicture(os.path.join(path, people_name, pic_name), d_size=32, color=PARA.GRAY,
                              normalization=PARA.MAX_32FLOAT)
            if train_tag == '0':
                tr_data.append(img)
                tr_lab.append(lab)
                train_num += 1
            else:
                ts_data.append(img)
                ts_lab.append(lab)
                test_num += 1
        ROOT_LOG.info("Model {}'s data was loaded (TR{}TS{})".format(self.model_name, train_num, test_num))
        self.data_p, self.data_l, self.data_pt, self.data_lt = tr_data, tr_lab, ts_data, ts_lab

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
                print("\t LF {}".format(self.get_variable(self.loss_function, feed_dict=test_feed)))
                print("\t AC {}".format(self.get_variable(self.accuracy, feed_dict=test_feed)))
                # print("\t Real {}".format(self.get_variable(self.real_label, feed_dict=test_feed)))
                # print("\t Result {}".format(self.get_variable(self.predict_label, feed_dict=test_feed)))
        return self.get_variable(self.accuracy, feed_dict=test_feed)

    def trainInDisk(self, step, batchsize):
        """
        To do
        :param step:
        :param batchsize:
        :return:
        """
        return self.get_variable(self.accuracy, feed_dict=test_feed)


if __name__ == "__main__":
    namelist = os.listdir(TR_PATH)
    print("Info: model_amount {}".format(len(namelist)))
    for i, model_name in enumerate(namelist):
        C1 = CNN1(input_size=[32, 32, 1], label_size=PEOPLE_SIZE, model_name=model_name)
        C1.build_model(False)
        C1.initial_para()
        C1.loadDataInMemory(os.path.join(TR_PATH, model_name), os.path.join(TR_CSV, str(i) + ".csv"),
                            number_limit=10000)
        AC = C1.trainInMemory(batchsize=100, step=50000)
        output = open(os.path.join(RE_PATH, "result.txt"), 'a')
        output.write("model {}: ac {} \n".format(model_name, AC))
        output.close()
        C1.save_para()
        del C1
