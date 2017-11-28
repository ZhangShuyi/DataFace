# -*- coding: utf-8 -*-
"""
    Version 1.0
    Sep batches
    Use for separating the picture to batches
"""

import pandas
import dlib
from matplotlib import pyplot as plt
from InternalModule.LogSetting import PRINT_LOG, RECORD_FILE, ROOT_LOG
from InternalModule.FilePicAPI import *
from InternalModule.Envs import *
import random

# face_rec_model_path = 'D:\\face project\\dlib_face\\dlib_face_recognition_resnet_model_v1.dat'
detector = dlib.get_frontal_face_detector()
LD_MODEL = dlib.shape_predictor(LD_MODEL_PATH)


def GetPicLandmarkDict(img):
    picture_diction = {}
    rect = dlib.rectangle(0, 0, img.shape[0], img.shape[1])
    try:
        shape = LD_MODEL(img, rect)
        for i in range(LANDMARK_SIZE):
            picture_diction["%d" % i] = [shape.part(i).x, shape.part(i).y]
    except KeyError:
        ROOT_LOG.warning("Get PicLandmark:Img have no face!")
        return None
    return picture_diction


def GetPicBatches(img, d_size, k_size, show=False):
    batches = []
    landmark_coordinate = GetPicLandmarkDict(img)
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


def GetPicBatchesV2(img, batch_index, show):
    def resize_image(img):
        return cv2.resize(img, dsize=(D_SIZE, D_SIZE), interpolation=cv2.INTER_CUBIC).reshape(D_SIZE, D_SIZE, 1)

    if batch_index > MODEL_NUM or batch_index < 0:
        ROOT_LOG.info("GetBatches: batch_index error")
        return
    landmark_coordinate = GetPicLandmarkDict(img)
    img = numpy.array(img)
    size = img.shape
    img = img.reshape(img.shape[0], img.shape[1], 1)
    if batch_index == 0:
        return resize_image(img)
    if batch_index == 1:
        return resize_image(img[:int(size[0] / 2), :, :])
    if batch_index == 2:
        return resize_image(img[int(size[0] / 2):, :, :])
    if batch_index == 3:
        return resize_image(img[:, :int(size[1] / 2), :])
    if batch_index == 4:
        return resize_image(img[:, int(size[1] / 2):, :])
    if batch_index == 5:
        return resize_image(img[:int(size[0] / 2), :int(size[1] / 2), :])
    if batch_index == 6:
        return resize_image(img[:int(size[0] / 2), int(size[1] / 2):, :])
    if batch_index == 7:
        return resize_image(img[int(size[0] / 2):, :int(size[1] / 2), :])
    if batch_index == 8:
        return resize_image(img[int(size[0] / 2):, int(size[1] / 2):, :])
    if batch_index > 8 and batch_index < MODEL_NUM:
        index = "%d" % batch_index - 9
        left = max(0, landmark_coordinate[index][0] - K_SIZE)
        right = min(63, landmark_coordinate[index][0] + K_SIZE)
        top = max(0, landmark_coordinate[index][1] - K_SIZE)
        bottom = min(63, landmark_coordinate[index][1] + K_SIZE)
        try:
            return resize_image(img[left:right, top:bottom, :])
        except cv2.error:
            ROOT_LOG.warning("A batch coordinate is invalid \n \t \
                    (left {}, right{}, top{} ,bottom{})".format(left, right, top, bottom))
            return None


# return {model_name1:{pic_name1: $batch,pic_name2:$batch},\
#         model_name2:{...},\
#         ...}
def GetPeopleBatchesDict(people_name):
    dict = {}
    for model in MODEL_LIST:
        dict[model] = {}
    img_dict = LoadPicsAsDict(pic_path=os.path.join(DATA_PATH, people_name), d_size=D_SIZE)
    for pic_name in img_dict:
        img = img_dict[pic_name]
        batches = GetPicBatches(img, D_SIZE, K_SIZE)
        for model in MODEL_LIST:
            model_index = int(model)
            dict[model][pic_name] = batches[model_index].tolist()
    return dict


# write {people1:{pic_nam1:$batch,pic_name2:$batch},\
#        people2:{...},\
#         ...}


def WriteAllBatches():
    def WriteTestBatches():
        dict = {}
        for model in MODEL_LIST:
            dict[model] = {}
        for index, people_name in enumerate(PEOPLE_TEST):
            people_batch = GetPeopleBatchesDict(people_name)
            for model_index, model in enumerate(MODEL_LIST):
                dict[model][people_name] = people_batch[model]
            PRINT_LOG.info("Get index{} people {}'s all batches".format(index, people_name))
        for model in MODEL_LIST:
            write_path = os.path.join(TEST_PATH, model + ".json")
            SavePicsAsJson(dict[model], write_path)
            PRINT_LOG.info("Write Test data Model {} Batches successfully".format(model))

    def WriteTrain1Batches():
        dict = {}
        for model in MODEL_LIST:
            dict[model] = {}
        for index, people_name in enumerate(PEOPLE_TRAIN1):
            people_batch = GetPeopleBatchesDict(people_name)
            for model_index, model in enumerate(MODEL_LIST):
                dict[model][people_name] = people_batch[model]
            PRINT_LOG.info("Get index{} people {}'s all batches".format(index, people_name))
        for model in MODEL_LIST:
            write_path = os.path.join(TRAIN_PATH1, model + ".json")
            SavePicsAsJson(dict[model], write_path)
            PRINT_LOG.info("Write Train1 data Model {} Batches successfully".format(model))

    def WriteTrain2Batches():
        dict = {}
        for model in MODEL_LIST:
            dict[model] = {}
        for index, people_name in enumerate(PEOPLE_TRAIN2):
            people_batch = GetPeopleBatchesDict(people_name)
            for model_index, model in enumerate(MODEL_LIST):
                dict[model][people_name] = people_batch[model]
            PRINT_LOG.info("Get index{} people {}'s all batches".format(index, people_name))
        for model in MODEL_LIST:
            write_path = os.path.join(TRAIN_PATH2, model + ".json")
            SavePicsAsJson(dict[model], write_path)
            PRINT_LOG.info("Write Train2 Model {} Batches successfully".format(model))

    WriteTestBatches()
    WriteTrain1Batches()
    WriteTrain2Batches()


if __name__ == "__main__":
    WriteAllBatches()
