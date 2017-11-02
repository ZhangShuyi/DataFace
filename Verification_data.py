import tensorflow as tf
import os
import json
import numpy
import random

DATA_TEST_PATH = "D:\\data1\\data3\\featureID_test"
DATA_PATH = "D:\\data1\\data3\\featureID"
WRITE_PATH = "D:\\data1\\data3\\writer2"
MODEL_LIST = [1, 51, 47, 2, 3]


class Veri_net:
    @staticmethod
    def add_full_connection_layer(input_tensor, in_dim, out_dim, stddev, name, active_function=None, board=False):
        with tf.name_scope(name):
            weight = tf.get_variable(name + "W", [in_dim, out_dim],
                                     initializer=tf.truncated_normal_initializer(stddev=stddev))
            basis = tf.get_variable(name + "B", [out_dim], initializer=tf.constant_initializer(0.0))
            output = tf.add(tf.matmul(input_tensor, weight), basis)
            if active_function:
                ac_output = active_function(output)
            else:
                ac_output = output
            if board:
                tf.summary.histogram(name + "\W", weight)
                tf.summary.histogram(name + "\B", basis)
                tf.summary.histogram(name + "\out", output)
                tf.summary.histogram(name + "\\ac_output", ac_output)
            return ac_output

    @staticmethod
    def add_normalization_layer(input_tensor, name, board=False):
        with tf.name_scope(name):
            output = tf.nn.l2_normalize(input_tensor, dim=1)
            if board:
                tf.summary.histogram("input", input_tensor)
                tf.summary.histogram("output", output)
        return output

    @staticmethod
    def lrelu(x, leak=0.2):
        return tf.maximum(x, x * leak)

    def __init__(self, model_list, model_feature_size, name):
        self.model_list = model_list
        self.model_num = len(model_list)
        self.model_feature_size = model_feature_size
        self.name = name
        self.sess = tf.Session()
        print("Info: Initial Verification net {}".format(self.name))

    def __del__(self):
        print("Info: Verification model was deleted")

    def build_model(self):
        dim_list = [self.model_feature_size, self.model_feature_size / 2, self.model_num * self.model_feature_size / 2,
                    2]
        self.input_x_1 = tf.placeholder(shape=[None, self.model_feature_size * self.model_num], dtype=tf.float32,
                                        name="input_x")
        self.input_x_2 = tf.placeholder(shape=[None, self.model_feature_size * self.model_num], dtype=tf.float32,
                                        name="input_x")
        self.real_label = tf.placeholder(shape=[None, 2], dtype=tf.float32, name="real_label")

        with tf.variable_scope("Verification"):
            layer1_1 = []
            for i in range(self.model_num):
                start = i * self.model_feature_size
                end = (i + 1) * self.model_feature_size
                layer_t = self.add_full_connection_layer(input_tensor=self.input_x_1[:, start:end], in_dim=dim_list[0],
                                                         out_dim=dim_list[1],
                                                         stddev=0.01, name="1_%02d" % i, active_function=tf.nn.softmax)
                layer1_1.append(layer_t)
            layer1_2 = []
            for i in range(self.model_num):
                start = i * self.model_feature_size
                end = (i + 1) * self.model_feature_size
                layer_t = self.add_full_connection_layer(input_tensor=self.input_x_2[:, start:end], in_dim=dim_list[0],
                                                         out_dim=dim_list[1],
                                                         stddev=0.01, name="2_%02d" % i, active_function=tf.nn.softmax)
                layer1_2.append(layer_t)
            layer1_1_set = []
            layer1_2_set = []
            for i in range(self.model_num):
                layer1_1_set.append(layer1_1[i])
                layer1_2_set.append(layer1_2[i])
            layer1_1_output = tf.concat(layer1_1_set, 1)
            layer1_2_output = tf.concat(layer1_2_set, 1)
            layer1_output = tf.concat([layer1_1_output, layer1_2_output], 1)

            layer2 = self.add_full_connection_layer(input_tensor=layer1_output,
                                                    in_dim=2 * dim_list[1] * self.model_num, out_dim=dim_list[2],
                                                    stddev=0.01, name="VER_l2", active_function=tf.nn.relu)
            self.predict_label = self.add_full_connection_layer(input_tensor=layer2, in_dim=dim_list[2],
                                                                out_dim=dim_list[3],
                                                                stddev=0.01, name="VER_l3",
                                                                active_function=tf.nn.softmax)
            self.cross_entropy = -tf.reduce_mean(
                tf.reduce_sum(self.real_label * tf.log(self.predict_label), reduction_indices=[1]))
            self.train_step = tf.train.AdamOptimizer(0.1, name="Train").minimize(self.cross_entropy)

            self.correct_prediction = tf.equal(tf.arg_max(self.real_label, 1), tf.arg_max(self.predict_label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, dtype="float"))
            # tf.summary.scalar("ce", self.cross_entropy)
            # tf.summary.scalar("ac", self.accuracy)
            self.merged = tf.summary.merge_all()
            if not os.path.exists(WRITE_PATH):
                os.mkdir(WRITE_PATH)
            if not os.path.exists(os.path.join(WRITE_PATH, self.name)):
                os.mkdir(os.path.join(WRITE_PATH, self.name))
            self.writer = tf.summary.FileWriter(os.path.join(WRITE_PATH, self.name), self.sess.graph)
            self.saver = tf.train.Saver()

    def train(self, step, batch_size):
        print("Info: Net {} train start({}step, {}batches)".format(self.name, step, batch_size))
        test_data, test_data_label = self.load_batch(self.test_data, batch_size=100)
        test_data_x1 = [x[0] for x in test_data]
        test_data_x2 = [x[1] for x in test_data]
        for i in range(step):
            data, data_label = self.load_batch(self.data, batch_size=batch_size)
            data_x1 = [x[0] for x in data]
            data_x2 = [x[1] for x in data]
            # print(numpy.array(data_x1).shape)
            # print(numpy.array(data_x2).shape)
            self.sess.run(self.train_step,
                          feed_dict={self.input_x_1: data_x1, self.input_x_2: data_x2, self.real_label: data_label})
            if i % 100 == 0:
                print("ac:{}".format(self.sess.run(self.accuracy,
                                                   feed_dict={self.input_x_1: test_data_x1,
                                                              self.input_x_2: test_data_x2,
                                                              self.real_label: test_data_label})))

    def initial_para(self):
        init = tf.initialize_all_variables()
        self.sess.run(init)

    def save_para(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        save_path = self.saver.save(self.sess, os.path.join(path, self.name))
        print("Info: save verification model at {}".format(save_path))

    def restore_para(self, path):
        self.saver.restore(self.sess, os.path.join(path, self.name))
        print("Info: restore verification model from {}".format(os.path.join(path, self.name)))

    def load_test_data(self):
        print("Info: load test data begin")
        self.test_data = {}
        for model_name in os.listdir(DATA_TEST_PATH):
            if int(model_name[:4]) not in self.model_list:
                continue
            with open(os.path.join(DATA_TEST_PATH, model_name), 'r')as file_object:
                contents = json.load(file_object)
            for contents_index in contents:
                index_string = contents_index.split(".")[0]
                pic_num = index_string[-4:]
                people_name = index_string[:-4]
                if self.test_data.get(people_name) is None:
                    self.test_data[people_name] = {}
                if self.test_data[people_name].get(pic_num) is None:
                    self.test_data[people_name][pic_num] = []
                self.test_data[people_name][pic_num] += contents[contents_index]
        print("Info: load test data end")

    def load_train_data(self):
        print("Info: load train data begin")
        self.data = {}
        for model_name in os.listdir(DATA_PATH):
            if int(model_name[:4]) not in self.model_list:
                continue
            with open(os.path.join(DATA_PATH, model_name), 'r')as file_object:
                contents = json.load(file_object)
            for contents_index in contents:
                index_string = contents_index.split(".")[0]
                pic_num = index_string[-4:]
                people_name = index_string[:-4]
                if self.data.get(people_name) is None:
                    self.data[people_name] = {}
                if self.data[people_name].get(pic_num) is None:
                    self.data[people_name][pic_num] = []
                self.data[people_name][pic_num] += contents[contents_index]
        print("Info: load train data end")

    def load_positive_sample(self, data):
        people_data = random.sample(data.items(), 1)[0][1]
        data = random.sample(people_data.items(), 2)
        return [data[0][1], data[1][1]]

    def load_negative_sample(self, data):
        people_data = random.sample(data.items(), 2)
        data1 = random.sample(people_data[0][1].items(), 1)[0][1]
        data2 = random.sample(people_data[1][1].items(), 1)[0][1]
        return [data1, data2]

    def load_batch(self, data, batch_size=50, positive_tag=[1.0, 0.0], negative_tag=[0.0, 1.0]):
        re_data = []
        re_tag = []
        for i in range(batch_size):
            re_data.append(self.load_positive_sample(data))
            re_tag.append(positive_tag)
            re_data.append(self.load_negative_sample(data))
            re_tag.append(negative_tag)
        # print(numpy.array(re_data).shape)
        # print(numpy.array(re_tag).shape)
        return re_data, re_tag


if __name__ == "__main__":
    V = Veri_net(model_list=MODEL_LIST, model_feature_size=100, name="Verification")
    V.build_model()
    V.initial_para()
    V.load_train_data()
    V.load_test_data()
    V.train(step=500000, batch_size=50)
