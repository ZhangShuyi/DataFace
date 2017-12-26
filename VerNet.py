from InternalModule.Envs import *
import tensorflow as tf
import os
import numpy
import random
import pandas
import pickle
import FeatureID
from InternalModule.LogSetting import PRINT_LOG, RECORD_LOG, ROOT_LOG

VER_TRAIN_SAMPLE = 100000
VER_TEST_SAMPLE = 20000


class VERNET:
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
                tf.summary.histogram(name + "_W", weight)
                tf.summary.histogram(name + "_B", basis)
                tf.summary.histogram(name + "_out", output)
                tf.summary.histogram(name + "_ac_output", ac_output)
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

    def __init__(self, model_feature_size, name):
        self.model_num = len(LIST_MODEL)
        self.model_feature_size = model_feature_size
        self.name = name
        self.sess = tf.Session()
        print("Info: Initial Verification net {}".format(self.name))

    def __del__(self):
        print("Info: Verification model was deleted")

    def initial_para(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def save_para(self):
        if not os.path.exists(PATH_LAYER2_MODEL):
            os.mkdir(PATH_LAYER2_MODEL)
        save_path = self.saver.save(self.sess, os.path.join(PATH_LAYER2_MODEL, self.name))
        print("Info: save verification model at {}".format(save_path))

    def restore_para(self):
        model_file = tf.train.latest_checkpoint(os.path.join(PATH_LAYER2_MODEL))
        self.saver.restore(self.sess, model_file)
        print("Info: restore model from {}".format(os.path.join(PATH_LAYER2_MODEL, self.name)))


    def get_featureID_tr(self, people_name, pic_name):
        featureID = []
        for index, model in enumerate(LIST_MODEL):
            featureID += self.hand_dict_tr[model][people_name][pic_name]
        if len(featureID) == self.model_feature_size * self.model_num:
            return featureID
        else:
            PRINT_LOG.warning("A bad feature ID")
            return None

    def get_featureID_ts(self, people_name, pic_name):
        featureID = []
        for index, model in enumerate(LIST_MODEL):
            featureID += self.hand_dict_ts[model][people_name][pic_name]
        if len(featureID) == self.model_feature_size * self.model_num:
            return featureID
        else:
            PRINT_LOG.warning("A bad feature ID")
            return None

    def build_test_model(self):
        '''
        dim_list = [self.model_feature_size, self.model_feature_size / 2, self.model_num * self.model_feature_size / 10,
                    2]
        '''
        dim_list = [self.model_feature_size, self.model_feature_size / 2, self.model_feature_size / 2, 2]
        self.input_x_1 = tf.placeholder(shape=[None, self.model_feature_size * self.model_num], dtype=tf.float32,
                                        name="input_x")
        self.input_x_2 = tf.placeholder(shape=[None, self.model_feature_size * self.model_num], dtype=tf.float32,
                                        name="input_x")
        self.real_label = tf.placeholder(shape=[None, 2], dtype=tf.float32, name="real_label")

        with tf.variable_scope("Verification"):
            self.l1_weight1 = tf.get_variable("l1_W1", [dim_list[0], dim_list[1]],
                                              initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.l1_basis1 = tf.get_variable("l1_B1", [dim_list[1]], initializer=tf.constant_initializer(0.0))
            self.l1_output1 = tf.nn.relu(tf.add(tf.matmul(self.input_x_1, self.l1_weight1), self.l1_basis1))

            self.l1_weight2 = tf.get_variable("l1_W2", [dim_list[0], dim_list[1]],
                                              initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.l1_basis2 = tf.get_variable("l1_B2", [dim_list[1]], initializer=tf.constant_initializer(0.0))
            self.l1_output2 = tf.nn.relu(tf.add(tf.matmul(self.input_x_2, self.l1_weight2), self.l1_basis2))

            self.l2_input = tf.concat([self.l1_output1, self.l1_output2], 1)

            self.l2_weight = tf.get_variable("l2_W", [2 * dim_list[1], 2 * dim_list[2]],
                                             initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.l2_basis = tf.get_variable("l2_B", [2 * dim_list[2]], initializer=tf.constant_initializer(0.0))
            self.l2_output = tf.nn.softmax(tf.add(tf.matmul(self.l2_input, self.l2_weight), self.l2_basis))

            self.l3_weight = tf.get_variable("l3_W", [2 * dim_list[2], dim_list[3]],
                                             initializer=tf.truncated_normal_initializer(stddev=0.01))
            self.l3_basis = tf.get_variable("l3_B", [dim_list[3]], initializer=tf.constant_initializer(0.0))
            self.l3_output = tf.nn.softmax(tf.add(tf.matmul(self.l2_output, self.l3_weight), self.l3_basis))

            # self.cross_entropy = -tf.reduce_sum(tf.)
            self.cross_entropy = -tf.reduce_mean(
                tf.reduce_sum(self.real_label * tf.log(self.l3_output), reduction_indices=[1]))
            self.train_step = tf.train.AdamOptimizer(0.0001, name="Train").minimize(self.cross_entropy)

            self.correct_prediction = tf.equal(tf.argmax(self.real_label, 1), tf.argmax(self.l3_output, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, dtype="float"))
            self.merged = tf.summary.merge_all()
            if not os.path.exists(PATH_BROAD):
                os.mkdir(PATH_BROAD)
            if not os.path.exists(os.path.join(PATH_BROAD, self.name)):
                os.mkdir(os.path.join(PATH_BROAD, self.name))
            self.writer = tf.summary.FileWriter(os.path.join(PATH_BROAD, self.name), self.sess.graph)
            self.saver = tf.train.Saver()

    def build_model(self):
        dim_list = [self.model_feature_size, self.model_feature_size / 2, self.model_num * self.model_feature_size / 2,
                    2]
        self.input_x_1 = tf.placeholder(shape=[None, self.model_feature_size * self.model_num], dtype=tf.float32,
                                        name="input_x1")
        self.input_x_2 = tf.placeholder(shape=[None, self.model_feature_size * self.model_num], dtype=tf.float32,
                                        name="input_x2")
        self.real_label = tf.placeholder(shape=[None, 2], dtype=tf.float32, name="real_label")

        with tf.variable_scope("Verification"):
            layer1_1 = []
            for i in range(self.model_num):
                start = i * self.model_feature_size
                end = (i + 1) * self.model_feature_size
                layer_t = self.add_full_connection_layer(input_tensor=self.input_x_1[:, start:end], in_dim=dim_list[0],
                                                         out_dim=dim_list[1],
                                                         stddev=0.01, name="1_%02d" % i, active_function=tf.nn.relu,
                                                         board=True)
                layer1_1.append(layer_t)
            layer1_2 = []
            for i in range(self.model_num):
                start = i * self.model_feature_size
                end = (i + 1) * self.model_feature_size
                layer_t = self.add_full_connection_layer(input_tensor=self.input_x_2[:, start:end], in_dim=dim_list[0],
                                                         out_dim=dim_list[1],
                                                         stddev=0.01, name="2_%02d" % i, active_function=tf.nn.relu,
                                                         board=True)
                layer1_2.append(layer_t)
            layer1_1_set = []
            layer1_2_set = []
            for i in range(self.model_num):
                layer1_1_set.append(layer1_1[i])
                layer1_2_set.append(layer1_2[i])
            layer1_1_output = tf.concat(layer1_1_set, 1)
            layer1_2_output = tf.concat(layer1_2_set, 1)
            layer1_output = tf.concat([layer1_1_output, layer1_2_output], 1)

            self.layer2 = self.add_full_connection_layer(input_tensor=layer1_output,
                                                         in_dim=2 * dim_list[1] * self.model_num, out_dim=dim_list[2],
                                                         stddev=0.01, name="VER_l2", active_function=tf.nn.relu,
                                                         board=True)
            self.predict_label = self.add_full_connection_layer(input_tensor=self.layer2, in_dim=dim_list[2],
                                                                out_dim=dim_list[3],
                                                                stddev=0.01, name="VER_l3",
                                                                active_function=tf.nn.softmax, board=True)
            self.cross_entropy = tf.reduce_sum(tf.square(tf.subtract(self.real_label[:, 0], self.predict_label[:, 0])))
            self.train_step = tf.train.AdamOptimizer(0.00001, name="Train").minimize(self.cross_entropy)

            self.correct_prediction = tf.equal(tf.argmax(self.real_label, 1), tf.argmax(self.predict_label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, dtype="float"))
            # tf.summary.scalar("ce", self.loss_function)
            # tf.summary.scalar("ac", self.accuracy)
            self.merged = tf.summary.merge_all()
            if not os.path.exists(PATH_BROAD):
                os.mkdir(PATH_BROAD)
            if not os.path.exists(os.path.join(PATH_BROAD, self.name)):
                os.mkdir(os.path.join(PATH_BROAD, self.name))
            self.writer = tf.summary.FileWriter(os.path.join(PATH_BROAD, self.name), self.sess.graph)
            self.saver = tf.train.Saver()

    def get_accuracy(self, batch_size):
        test_data, test_data_label = self.load_batch("TS", batch_size=batch_size)
        test_data_x1 = [x[0] for x in test_data]
        test_data_x2 = [x[1] for x in test_data]
        return self.sess.run(self.accuracy,
                             feed_dict={self.input_x_1: test_data_x1, self.input_x_2: test_data_x2,
                                        self.real_label: test_data_label})

    def train(self, step, batch_size, DEBUG=False):
        print("Info: Net {} train start({} step, {} batches size)".format(self.name, step, batch_size))
        test_data_x1 = [x[0] for x in self.ts_data]
        test_data_x2 = [x[1] for x in self.ts_data]
        test_data_label = [x[2] for x in self.ts_data]
        sample_size = len(self.tr_data)
        for i in range(step):
            train_start = (batch_size * i) % sample_size
            train_end = (batch_size * (i + 1)) % sample_size
            if train_end < train_start:
                continue
            data_x1 = [x[0] for x in self.tr_data[train_start:train_end]]
            data_x2 = [x[1] for x in self.tr_data[train_start:train_end]]
            data_label = [x[2] for x in self.tr_data[train_start:train_end]]
            self.sess.run(self.train_step,
                          feed_dict={self.input_x_1: data_x1, self.input_x_2: data_x2, self.real_label: data_label})

            if i % 300 == 0:
                print("Step:{}".format(i))

                print("\t ac:{}".format(self.sess.run(self.accuracy,
                                                      feed_dict={self.input_x_1: test_data_x1,
                                                                 self.input_x_2: test_data_x2,
                                                                 self.real_label: test_data_label})))
                print("\t ce:{}".format(self.sess.run(self.cross_entropy,
                                                      feed_dict={self.input_x_1: test_data_x1,
                                                                 self.input_x_2: test_data_x2,
                                                                 self.real_label: test_data_label})))
                print("\t T_ac:{}".format(self.sess.run(self.accuracy,
                                                        feed_dict={self.input_x_1: data_x1,
                                                                   self.input_x_2: data_x2,
                                                                   self.real_label: data_label})))
            if i % 10000 == 0:
                self.save_para()

    # sample [identity1, pic1, identity2, pic2, flag]
    def load_sample(self):
        path = FILE_SAMPLE_LAYER2 + '_' + str(VER_TRAIN_SAMPLE) + '_' + str(VER_TEST_SAMPLE)
        if os.path.exists(path):
            with open(path, 'rb') as file:
                data = pickle.load(file)
            self.tr_sample, self.ts_sample = data
        else:
            self.tr_sample = []
            self.ts_sample = []
            for index in range(VER_TRAIN_SAMPLE):
                # positive
                while (True):
                    identity = random.sample(LIST_TRAIN2_PEOPLE, 1)[0]
                    frame = FRAME_MERGE_DATA[FRAME_MERGE_DATA.identity == identity]
                    pic_list = list(frame.pic_name)
                    if (len(pic_list) < 2):
                        continue
                    else:
                        pic_name = random.sample(pic_list, 2)
                        self.tr_sample.append([identity, pic_name[0], identity, pic_name[1], 1])
                        break
                # negative
                identity = random.sample(LIST_TRAIN2_PEOPLE, 2)
                frame1 = FRAME_MERGE_DATA[FRAME_MERGE_DATA.identity == identity[0]]
                pic_list1 = list(frame1.pic_name)
                pic_name1 = random.sample(pic_list1, 1)[0]
                frame2 = FRAME_MERGE_DATA[FRAME_MERGE_DATA.identity == identity[1]]
                pic_list2 = list(frame2.pic_name)
                pic_name2 = random.sample(pic_list2, 1)[0]
                self.tr_sample.append([identity[0], pic_name1, identity[1], pic_name2, 0])

            for index in range(VER_TEST_SAMPLE):
                # positive
                while (True):
                    identity = random.sample(LIST_TEST_PEOPLE, 1)[0]
                    frame = FRAME_MERGE_DATA[FRAME_MERGE_DATA.identity == identity]
                    pic_list = list(frame.pic_name)
                    if (len(pic_list) < 2):
                        continue
                    else:
                        pic_name = random.sample(pic_list, 2)
                        self.ts_sample.append([identity, pic_name[0], identity, pic_name[1], 1])
                        break
                # negative
                identity = random.sample(LIST_TEST_PEOPLE, 2)
                frame1 = FRAME_MERGE_DATA[FRAME_MERGE_DATA.identity == identity[0]]
                pic_list1 = list(frame1.pic_name)
                pic_name1 = random.sample(pic_list1, 1)[0]
                frame2 = FRAME_MERGE_DATA[FRAME_MERGE_DATA.identity == identity[1]]
                pic_list2 = list(frame2.pic_name)
                pic_name2 = random.sample(pic_list2, 1)[0]
                self.ts_sample.append([identity[0], pic_name1, identity[1], pic_name2, 0])

            with open(path, 'wb') as file:
                pickle.dump((self.tr_sample, self.ts_sample), file)
        return

    def load_featureID(self):
        self.tr_data = []
        self.ts_data = []
        random.shuffle(self.tr_sample)
        random.shuffle(self.ts_sample)
        F_tr = FeatureID.FeatureID(False)
        for line in self.tr_sample:
            featureID1 = F_tr.GetAPicWholeID(line[0], line[1])
            featureID2 = F_tr.GetAPicWholeID(line[2], line[3])
            tag = numpy.zeros([2])
            tag[line[4]] = 1.0
            self.tr_data.append([featureID1, featureID2, tag])
        F_ts = FeatureID.FeatureID(True)
        for line in self.ts_sample:
            featureID1 = F_ts.GetAPicWholeID(line[0], line[1])
            featureID2 = F_ts.GetAPicWholeID(line[2], line[3])
            tag = numpy.zeros([2])
            tag[line[4]] = 1.0
            self.ts_data.append([featureID1, featureID2, tag])


if __name__ == "__main__":
    V = VERNET(model_feature_size=TRAIN_PARA_FEATURE_ID_LEN, name="Verification")
    V.load_sample()
    V.load_featureID()
    V.build_model()
    V.initial_para()
    V.train(step=300000, batch_size=100, DEBUG=False)
