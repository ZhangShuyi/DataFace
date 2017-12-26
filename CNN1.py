import tensorflow as tf
import random
from PicToData import *
from BatchData import *
from Layer1DataProcess import *


class CNN1(object):
    @staticmethod
    def STATIC_lrelu(x, leak=0.2):
        return tf.maximum(x, x * leak)

    @staticmethod
    def STATIC_add_convolution_layer(image, input_channel, filter_size, stddev, output_channel, strides, name,
                                     active_function=None, record=False, keep_prob=1.0):
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
            ac_output = tf.nn.dropout(ac_output, keep_prob=keep_prob)
            return ac_output

    @staticmethod
    def STATIC_add_pooling_layer(image, ksize, strides, name, record=False):
        with tf.name_scope(name):
            pooling = tf.nn.max_pool(image, ksize, strides, padding='SAME')
            if record:
                tf.summary.histogram(name + "\pooling", pooling)
        return pooling

    @staticmethod
    def STATIC_add_full_connection_layer(input_tensor, in_dim, out_dim, stddev, name, active_function=None,
                                         record=False,
                                         keep_prob=1.0):
        with tf.name_scope(name):
            weight = tf.get_variable(name + "W", [in_dim, out_dim],
                                     initializer=tf.truncated_normal_initializer(stddev=stddev))
            basis = tf.get_variable(name + "B", [out_dim], initializer=tf.constant_initializer(0.0))
            export = tf.add(tf.matmul(input_tensor, weight), basis)
            if record:
                tf.summary.histogram(name + "\W", weight)
                tf.summary.histogram(name + "\B", basis)
                tf.summary.histogram(name + "\out", export)
            export = tf.nn.dropout(export, keep_prob=keep_prob)
            if active_function:
                return active_function(export)
            else:
                return export

    @staticmethod
    def STATIC_add_full_connection_layer_with_w(input_tensor, in_dim, out_dim, stddev, name, active_function=None,
                                                record=False,
                                                keep_prob=1.0):
        with tf.name_scope(name):
            weight = tf.get_variable(name + "W", [in_dim, out_dim],
                                     initializer=tf.truncated_normal_initializer(stddev=stddev))
            basis = tf.get_variable(name + "B", [out_dim], initializer=tf.constant_initializer(0.0))
            export = tf.add(tf.matmul(input_tensor, weight), basis)
            if record:
                tf.summary.histogram(name + "\W", weight)
                tf.summary.histogram(name + "\B", basis)
                tf.summary.histogram(name + "\out", export)
            export = tf.nn.dropout(export, keep_prob=keep_prob)
            w_2_n = tf.reduce_sum(tf.square(weight)) / out_dim
            if active_function:
                return active_function(export), w_2_n
            else:
                return export, w_2_n

    @staticmethod
    def STAITC_add_normalization_layer(input_tensor, name, record=False):
        with tf.name_scope(name):
            export = tf.nn.l2_normalize(input_tensor, dim=1)
            if record:
                tf.summary.histogram("input", input_tensor)
                tf.summary.histogram("output", export)
        return export

    def __init__(self, input_size, label_size, model_name, feature_id_length, train_keep_prob, device_num='/gpu:0'):
        try:
            self.input_height = input_size[0]
            self.input_width = input_size[1]
            self.input_channel = input_size[2]
        except KeyError:
            ROOT_LOG.error("Model {}'s input size is invalid".format(model_name))
            return
        self.model_name = model_name
        self.label_size = label_size
        self.feature_id_length = feature_id_length
        self.sess = tf.Session()
        self.train_keep_prob = train_keep_prob
        self.device_num = device_num
        ROOT_LOG.info("Model {} init success".format(self.model_name))

    def build_model(self, use_flag, record=False):
        with tf.device(self.device_num):
            self.image_size = [self.input_height, self.input_width, self.input_channel]
            self.real_label = tf.placeholder(tf.float32, shape=[None, self.label_size], name='Real_Label')
            self.image_input = tf.placeholder(tf.float32, shape=[None] + self.image_size, name='Image')
            self.keep_prob = tf.placeholder(tf.float32)

            with tf.variable_scope(self.model_name, reuse=use_flag):
                # neural structure
                channel_size_list = [self.input_channel, 20, 40, 60, 80]
                c1 = self.STATIC_add_convolution_layer(self.image_input, input_channel=channel_size_list[0],
                                                       output_channel=channel_size_list[1], filter_size=[3, 3],
                                                       stddev=0.2,
                                                       strides=[1, 1, 1, 1],
                                                       active_function=self.STATIC_lrelu,
                                                       name="con_l1",
                                                       keep_prob=self.keep_prob)
                p1 = self.STATIC_add_pooling_layer(c1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name="pool_l1")
                # size [16,16,20]

                c2 = self.STATIC_add_convolution_layer(p1, input_channel=channel_size_list[1],
                                                       output_channel=channel_size_list[2],
                                                       filter_size=[3, 3], stddev=0.02,
                                                       strides=[1, 1, 1, 1],
                                                       active_function=self.STATIC_lrelu,
                                                       name="con_l2",
                                                       keep_prob=self.keep_prob)
                p2 = self.STATIC_add_pooling_layer(c2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name="pool_l2")
                # size [8,8,40]

                c3 = self.STATIC_add_convolution_layer(p2, input_channel=channel_size_list[2], filter_size=[3, 3],
                                                       stddev=0.02,
                                                       output_channel=channel_size_list[3], strides=[1, 1, 1, 1],
                                                       active_function=self.STATIC_lrelu, name="con_l3",
                                                       keep_prob=self.keep_prob)
                p3 = self.STATIC_add_pooling_layer(c3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name="pool_l3")
                # size [4,4 60]

                c4 = self.STATIC_add_convolution_layer(p3, input_channel=channel_size_list[3], filter_size=[3, 3],
                                                       stddev=0.02,
                                                       output_channel=channel_size_list[4], strides=[1, 1, 1, 1],
                                                       name="con_l4",
                                                       active_function=self.STATIC_lrelu,
                                                       keep_prob=self.keep_prob)

                p4 = self.STATIC_add_pooling_layer(c4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], name="pool_l4")
                # size [2,2,80]

                # full connection layer
                p3_size = int(DATA_PARA_D_SIZE / 8)
                p4_size = int(DATA_PARA_D_SIZE / 16)
                p3_flat = tf.reshape(p3, [-1, p3_size * p3_size * 60])
                p4_flat = tf.reshape(p4, [-1, p4_size * p4_size * 80])

                f1 = tf.concat([p3_flat, p4_flat], 1)
                nf1 = self.STAITC_add_normalization_layer(f1, name="normalization")

                self.feature = self.STATIC_add_full_connection_layer(nf1,
                                                                     p3_size * p3_size * 60 + p4_size * p4_size * 80,
                                                                     out_dim=self.feature_id_length,
                                                                     stddev=0.1,
                                                                     name="feature",
                                                                     active_function=self.STATIC_lrelu,
                                                                     keep_prob=self.keep_prob)

                self.predict_label, self.w_n = self.STATIC_add_full_connection_layer_with_w(self.feature,
                                                                                            in_dim=self.feature_id_length,
                                                                                            out_dim=self.label_size,
                                                                                            stddev=0.1,
                                                                                            name="label",
                                                                                            active_function=tf.nn.softmax)
                self.loss_function = tf.reduce_sum(
                    tf.square(tf.subtract(self.real_label, self.predict_label))) + self.w_n * TRAIN_PARA_WN_LAMBDA

                # self.loss_function = tf.reduce_sum(tf.square(tf.subtract(self.real_label, self.predict_label)))

                # self.loss_function = -tf.reduce_mean(
                #    tf.reduce_sum(self.real_label * tf.log(self.predict_label), reduction_indices=[1]))
                self.train_step = tf.train.AdamOptimizer(0.0001, name="Train").minimize(self.loss_function)

            self.correct_prediction = tf.equal(tf.argmax(self.real_label, 1), tf.argmax(self.predict_label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, dtype="float"))
            if record:
                tf.summary.scalar("AC", self.accuracy)
                tf.summary.scalar("loss", self.loss_function)
                self.merged = tf.summary.merge_all()
                if not os.path.exists(os.path.join(PATH_BROAD, self.model_name)):
                    os.mkdir(os.path.join(PATH_BROAD, self.model_name))
                self.writer = tf.summary.FileWriter(os.path.join(PATH_BROAD, self.model_name), self.sess.graph)

        self.saver = tf.train.Saver()

    def para_initial(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def para_save(self):
        if not os.path.exists(os.path.join(PATH_LAYER1_MODEL, self.model_name)):
            os.mkdir(os.path.join(PATH_LAYER1_MODEL, self.model_name))
        save_path = self.saver.save(self.sess, os.path.join(PATH_LAYER1_MODEL, self.model_name, self.model_name))
        print("Info: save model at {}".format(save_path))

    def para_restore(self):
        print(os.path.join(PATH_LAYER1_MODEL, self.model_name))
        model_file = tf.train.latest_checkpoint(os.path.join(PATH_LAYER1_MODEL, self.model_name))
        self.saver.restore(self.sess, model_file)
        print("Info: restore model from {} successfully".format(
            os.path.join(PATH_LAYER1_MODEL, self.model_name, self.model_name)))

    def variable_get(self, variable, feed_dict):
        return self.sess.run(variable, feed_dict=feed_dict)

    def variable_write(self, index, feed_dict):
        result = self.sess.run(self.merged, feed_dict=feed_dict)
        self.writer.add_summary(result, index)

    def get_feature(self, image):
        if image is None:
            return numpy.zeros(self.feature_id_length)
        return self.sess.run(self.feature,
                             feed_dict={self.image_input: [image], self.keep_prob: 1.0, self.real_label: [
                                 numpy.zeros(self.label_size)]})[0]

    def get_identity(self, image):
        if image is None:
            return None
        else:
            return self.sess.run(tf.argmax(self.predict_label, 1),
                                 feed_dict={self.image_input: [image],
                                            self.real_label: [numpy.zeros(self.label_size)]})

    def check_accuracy(self):
        ts = []
        cross_operation = CrossValidationOperation(10, LIST_PEOPLE_LAYER1, "layer1_1000_10")
        cross_operation.restore_cross_validation()
        for people_index, people_name in enumerate(LIST_PEOPLE_LAYER1):
            lab = numpy.zeros(self.label_size)
            lab[people_index] = 1.0
            sample_number = 0
            data = OneBatch(self.model_name, LIST_TRAIN1_PEOPLE, 2).loadOneIdentity(people_name)
            for pic_name in data:
                sample_number += 1
                b = data[pic_name]
                # b, r, g = cv2.split(data[pic_name])
                img = numpy.array(b).reshape(DATA_PARA_D_SIZE, DATA_PARA_D_SIZE, 1)
                if cross_operation.search_cross_validation(pic_name) in [1, 2]:
                    ts.append([img, lab])
        random.shuffle(ts)
        ts_data = [x[0] for x in ts]
        ts_lab = [x[1] for x in ts]
        test_feed = {self.image_input: ts_data, self.real_label: ts_lab, self.keep_prob: 1.0}
        print("model {} ac{}".format(self.model_name, self.variable_get(self.accuracy, feed_dict=test_feed)))

    def data_structureFromDisk(self):
        tr = []
        ts = []
        cross_operation = CrossValidationOperation(10, LIST_PEOPLE_LAYER1, "layer1_1000_10")
        cross_operation.restore_cross_validation()
        for people_index, people_name in enumerate(LIST_PEOPLE_LAYER1):
            lab = numpy.zeros(self.label_size)
            lab[people_index] = 1.0
            sample_number = 0
            # pickle_operation = PickleOperation(save_path=PATH_PICKLE_STYLE, D_size=DATA_PARA_D_SIZE)
            # data = pickle_operation.ReadAPickle(people_name)
            data = OneBatch(self.model_name, LIST_TRAIN1_PEOPLE, 2).loadOneIdentity(people_name)
            for pic_name in data:
                sample_number += 1
                b = data[pic_name]
                # b, r, g = cv2.split(data[pic_name])
                img = numpy.array(r).reshape(DATA_PARA_D_SIZE, DATA_PARA_D_SIZE, 1)
                if cross_operation.search_cross_validation(pic_name) in [1, 2]:
                    ts.append([img, lab])
                else:
                    tr.append([img, lab])
            PRINT_LOG.info(
                "Load {} {} picture for model {} (total sample{})".format(people_index, people_name, self.model_name,
                                                                          sample_number))
        random.shuffle(ts)
        random.shuffle(tr)
        tr_data = [x[0] for x in tr]
        tr_lab = [x[1] for x in tr]
        ts_data = [x[0] for x in ts]
        ts_lab = [x[1] for x in ts]
        ROOT_LOG.info(
            "Model {}'s data was loaded (TR:{} TS:{})".format(self.model_name, len(tr_data), len(ts_data)))
        self.data_p, self.data_l, self.data_pt, self.data_lt = tr_data, tr_lab, ts_data, ts_lab

    def train_InDisk(self, step, batch_size):
        test_feed = {self.image_input: self.data_pt, self.real_label: self.data_lt, self.keep_prob: 1.0}
        sample_size = len(self.data_p)
        for i in range(step):
            train_start = (batch_size * i) % sample_size
            train_end = (batch_size * (i + 1)) % sample_size
            if train_end < train_start:
                continue
            # train_feed = {self.image_input: self.data_p,
            #              self.real_label: self.data_l}
            train_feed = {self.image_input: self.data_p[train_start:train_end],
                          self.real_label: self.data_l[train_start:train_end],
                          self.keep_prob: self.train_keep_prob}
            train_test_feed = {self.image_input: self.data_p[train_start:train_end],
                               self.real_label: self.data_l[train_start:train_end],
                               self.keep_prob: 1.0}
            self.sess.run(self.train_step, feed_dict=train_feed)

            if i % 500 == 0:
                print("\t Step {}".format(i))
                # print("\t LF {}".format(self.get_variable(self.loss_function, feed_dict=test_feed)))
                print("\t AC {}".format(self.variable_get(self.accuracy, feed_dict=test_feed)))
                # print("\t TLF {}".format(self.get_variable(self.loss_function, feed_dict=train_feed)))
                print("\t TAC {}".format(self.variable_get(self.accuracy, feed_dict=train_test_feed)))
            if i % 10000 == 0:
                self.para_save()
                output = open(os.path.join(PATH_RESULT, "result.txt"), 'a')
                output.write("model {} step {}  ac {} \n".format(model_name, i,
                                                                 self.variable_get(self.accuracy, feed_dict=test_feed)))
                output.close()
        return self.variable_get(self.accuracy, feed_dict=test_feed)


if __name__ == "__main__":
    # for i, model_name in enumerate(LIST_MODEL):
    #     print(DATA_PARA_PEOPLE_LIMIT_TRAIN1)
    #     print(len(LIST_PEOPLE_LAYER1))
    #     C1 = CNN1(input_size=[DATA_PARA_D_SIZE, DATA_PARA_D_SIZE, 1], label_size=DATA_PARA_PEOPLE_LIMIT_TRAIN1,
    #               model_name=model_name,
    #               feature_id_length=TRAIN_PARA_FEATURE_ID_LEN, train_keep_prob=0.5)
    #     C1.build_model(False)
    #     # C1.para_restore()
    #     C1.para_initial()
    #     C1.data_structureFromDisk()
    #     AC = C1.train_InDisk(batch_size=100, step=850000)
    C1 = CNN1(input_size=[DATA_PARA_D_SIZE, DATA_PARA_D_SIZE, 1], label_size=DATA_PARA_PEOPLE_LIMIT_TRAIN1,
              model_name='batch0_64_64_3',
              feature_id_length=TRAIN_PARA_FEATURE_ID_LEN, train_keep_prob=0.5)
    C1.build_model(False)
    C1.para_restore()
    C1.check_accuracy()
