# -*- coding: utf-8 -*-
"""
    Version 1.0
    Sep batches
    Use for separating the picture to batches
"""
import pandas
from matplotlib import pyplot as plt
from InternalModule.Header import *
from InternalModule.LogSetting import *
import random

PATH = "D:\\data1\\data3\\datas_face_train"
JSON_PATH = "D:\\data1\\data3\\datas_face_train_json"
FEATURE_PATH = "D:\\data1\\data3\\datas_face_train_feature"

T_PATH = "D:\\data1\\data3\\datas_face_test"
T_JSON_PATH = "D:\\data1\\data3\\datas_face_test_json"
T_FEATURE_PATH = "D:\\data1\\data3\\datas_face_test_feature"

LD_MODEL_PATH = 'D:\\face project/ModelAndTxt/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'D:\\face project\\dlib_face\\dlib_face_recognition_resnet_model_v1.dat'
detector = dlib.get_frontal_face_detector()
#        n   LD_MODEL = dlib.shape_predictor(LD_MODEL_PATH)
LANDMARK_SIZE = 68
D_SIZE = 64
K_SIZE = 10
TEST_PRO = 0.2
feature_map = {}


def GetPicLandmarkDict(img, img_name="None Name"):
    picture_diction = {}
    rect = dlib.rectangle(0, 0, img.shape[0], img.shape[1])
    try:
        shape = LD_MODEL(img, rect)
        for i in range(LANDMARK_SIZE):
            picture_diction["%d" % i] = [shape.part(i).x, shape.part(i).y]
    except KeyError:
        ROOT_LOG.warning("Img {} have no face!".format(img_name))
        return None
    return picture_diction


def GetPeopleLandmarkDict(img_dict, people_name=None, csv_path=None):
    people_diction = {}
    for img_index in img_dict:
        people_diction[img_index] = GetPicLandmarkDict(img_dict[img_index], img_index)
    csv_data = pandas.DataFrame(people_diction)
    if csv_path is not None:
        csv_data.to_csv(os.path.join(csv_path, people_name + ".csv"), 'w')
    return csv_data


def GetAllPeopleLandmarkCSV(path, csv_path):
    if not os.path.exists(csv_path):
        os.mkdir(csv_path)
    for people_name in os.listdir(path):
        pic_data = LoadFilePicInMemoryWithAddress(os.path.join(path, people_name), postfix='tif')
        GetPeopleLandmarkDict(pic_data, people_name, csv_path)
    ROOT_LOG.info("Write All landmark to path {}".format(csv_path))


# GetPicBatches V1.0
# 78 batches= 68 landmark + 10 slice
def GetPicBatches(img, d_size, k_size, landmark_coordinate, show=False):
    batches = []
    img = numpy.array(img)
    size = img.shape
    img = img.reshape(img.shape[0], img.shape[1], 1)
    batches.append(cv2.resize(img, dsize=(d_size, d_size), interpolation=cv2.INTER_CUBIC).reshape(d_size, d_size, 1))
    batches.append(
        cv2.resize(img[:int(size[0] / 2), :, :], dsize=(d_size, d_size), interpolation=cv2.INTER_CUBIC).reshape(d_size,
                                                                                                                d_size,
                                                                                                                1))
    batches.append(
        cv2.resize(img[:int(size[0] / 2), :, :], dsize=(d_size, d_size), interpolation=cv2.INTER_CUBIC).reshape(d_size,
                                                                                                                d_size,
                                                                                                                1))
    batches.append(
        cv2.resize(img[:, :int(size[1] / 2), :], dsize=(d_size, d_size), interpolation=cv2.INTER_CUBIC).reshape(d_size,
                                                                                                                d_size,
                                                                                                                1))
    batches.append(
        cv2.resize(img[:, int(size[1] / 2):, :], dsize=(d_size, d_size), interpolation=cv2.INTER_CUBIC).reshape(d_size,
                                                                                                                d_size,
                                                                                                                1))
    # 32*32 1/4 4
    batches.append(
        cv2.resize(img[:int(size[0] / 2), :int(size[1] / 2), :], dsize=(d_size, d_size),
                   interpolation=cv2.INTER_CUBIC).reshape(d_size, d_size, 1))
    batches.append(
        cv2.resize(img[:int(size[0] / 2), int(size[1] / 2):, :], dsize=(d_size, d_size),
                   interpolation=cv2.INTER_CUBIC).reshape(d_size, d_size, 1))
    batches.append(
        cv2.resize(img[int(size[0] / 2):, :int(size[1] / 2), :], dsize=(d_size, d_size),
                   interpolation=cv2.INTER_CUBIC).reshape(d_size, d_size, 1))
    batches.append(
        cv2.resize(img[int(size[0] / 2):, int(size[1] / 2):, :], dsize=(d_size, d_size),
                   interpolation=cv2.INTER_CUBIC).reshape(d_size, d_size, 1))

    # feature map
    for i, index in enumerate(landmark_coordinate):
        left = max(0, landmark_coordinate[index][0] - k_size)
        right = min(63, landmark_coordinate[index][0] + k_size)
        top = max(0, landmark_coordinate[index][1] - k_size)
        bottom = min(63, landmark_coordinate[index][1] + k_size)
        try:
            batches.append(
                cv2.resize(img[left:right, top:bottom, :], dsize=(d_size, d_size),
                           interpolation=cv2.INTER_CUBIC).reshape(d_size, d_size, 1))
        except cv2.error:
            ROOT_LOG.warning("A batch coordinate is invalid \n \t \
            (left {}, right{}, top{} ,bottom{})".format(left, right, top, bottom))
            batches.append(None)
    if show:
        for i, pic in enumerate(batches):
            plt.subplot(10, 10, i + 1)
            plt.axis('off')
            plt.imshow(batches[i].reshape(64, 64))
        plt.show()
    return batches


'''
def GetPeopleBatches(people_path, people_name, write_path):
    for pic_name in os.listdir(people_path):
        img = ReadPicture(os.path.join(people_path, pic_name), d_size=64)
        landmark_coordinate = GetPicLandmarkDict(img)
        batches = GetPicBatches(img, d_size=64, k_size=10, landmark_coordinate=landmark_coordinate)
        model_name_list = [model_name for model_name in range(len(batches))]
        for i, model_name in enumerate(model_name_list):
            if not os.path.exists(write_path, model_name):
                os.mkdir(os.path.join(write_path, model_name))
            if not os.path.exists(write_path, model_name, people_name):
                os.mkdir(os.path.join(write_path, model_name, people_name))
            WritePicture(batches[i], os.path(write_path, model_name, people_name, pic_name + ".jpg"))
    return
'''


# set d_size
# set k_size
def WritePeopleBatchesFile(people_path, people_name, write_path):
    for pic_name in os.listdir(people_path):
        img = ReadPicture(os.path.join(people_path, pic_name), to_array=False)
        landmark_coordinate = GetPicLandmarkDict(img)
        batches = GetPicBatches(img, d_size=D_SIZE, k_size=K_SIZE, landmark_coordinate=landmark_coordinate)
        model_name_list = [str(model_name) for model_name in range(len(batches))]
        for i, model_name in enumerate(model_name_list):
            if not os.path.exists(os.path.join(write_path, model_name)):
                os.mkdir(os.path.join(write_path, model_name))
            if not os.path.exists(os.path.join(write_path, model_name, people_name)):
                os.mkdir(os.path.join(write_path, model_name, people_name))
            WritePicture(batches[i], os.path.join(write_path, model_name, people_name, pic_name + ".jpg"))
    ROOT_LOG.info("Write People {} Batches File at {} success".format(people_name, write_path))
    return


def WriteAllBatchesFile(path, write_path):
    for people_index, people_name in enumerate(os.listdir(path)):
        WritePeopleBatchesFile(os.path.join(path, people_name), people_name, write_path)


# SET TEST_PRO
def GetModelAddressCSV(path, model_name, csv_path, log=False):
    csv_data = pandas.DataFrame(columns=('people_name', 'pic_name', 'Train/Test'))
    csv_index = 0
    ROOT_LOG.info("Start to generate Model{}'s Sources address".format(model_name))
    for people_name in os.listdir(path):
        train_num = 0
        test_num = 0
        pic_name_list = os.listdir(os.path.join(path, people_name))
        for pic_name in pic_name_list:
            test_flag = random.random()
            if test_flag > TEST_PRO:
                csv_data.loc[csv_index] = [people_name, pic_name, 0]
                train_num += 1
            else:
                csv_data.loc[csv_index] = [people_name, pic_name, 1]
                test_num += 1
            csv_index += 1
        if log:
            ROOT_LOG.info("people {} address generation finished \n \t \
                                with test {} train{}".format(people_name, test_num, train_num))
    csv_data.to_csv(csv_path, 'w')
    ROOT_LOG.info("Success generate model model{}'s  Address".format(model_name))


def GetAllAddressCSV(path, csv_file):
    for model_name in os.listdir(path):
        GetModelAddressCSV(os.path.join(path, model_name), model_name, os.path.join(csv_file, model_name + ".csv"))


if __name__ == "__main__":
    # GetAllPeopleLandmark("F:\\CAS-PEAL-R1-64_64", "D:\\data_csv")
    # WriteAllBatchesFile("F:\\CAS-PEAL-R1-64_64", "F:\\Feature")
    GetAllAddressCSV("F:\\Feature", "F:\\CSV")
