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



class DILBOperation:
    def __init__(self):
        self.global_DETECTOR = dlib.get_frontal_face_detector()
        self.global_LD_MODEL = dlib.shape_predictor(FILE_LD_MODEL)
        self.global_MODEL_NUMBER_BY_STEP = 10
        self.global_LANDMARK_SIZE = 68

    def GetPicLandmarkDict(self, img):
        picture_diction = {}
        rect = dlib.rectangle(0, 0, img.shape[0], img.shape[1])
        try:
            shape = self.global_LD_MODEL(img, rect)
            for i in range(self.global_LANDMARK_SIZE):
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
        return cv2.resize(img, dsize=(DATA_PARA_D_SIZE, DATA_PARA_D_SIZE), interpolation=cv2.INTER_CUBIC).reshape(
            DATA_PARA_D_SIZE, DATA_PARA_D_SIZE, 1)

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
        left = max(0, landmark_coordinate[index][0] - DATA_PARA_K_SIZE)
        right = min(63, landmark_coordinate[index][0] + DATA_PARA_K_SIZE)
        top = max(0, landmark_coordinate[index][1] - DATA_PARA_K_SIZE)
        bottom = min(63, landmark_coordinate[index][1] + DATA_PARA_K_SIZE)
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
    for model in LIST_MODEL:
        dict[model] = {}
    img_dict = LoadPicsAsDict(pic_path=os.path.join(PATH_DATA_ALIGNED, people_name), d_size=DATA_PARA_D_SIZE)
    for pic_name in img_dict:
        img = img_dict[pic_name]
        batches = GetPicBatches(img, DATA_PARA_D_SIZE, DATA_PARA_K_SIZE)
        for model in LIST_MODEL:
            model_index = int(model)
            if batches[model_index] is None:
                dict[model][pic_name] = None
            else:
                dict[model][pic_name] = batches[model_index].tolist()
    return dict


# write {people1:{pic_nam1:$batch,pic_name2:$batch},\
#        people2:{...},\
#         ...}


def WriteAllBatches():
    def WriteTestBatches():
        dict = {}
        for model in LIST_MODEL:
            dict[model] = {}
        for index, people_name in enumerate(PEOPLE_TEST):
            people_batch = GetPeopleBatchesDict(people_name)
            for model_index, model in enumerate(LIST_MODEL):
                dict[model][people_name] = people_batch[model]
            PRINT_LOG.info("Get index{} people {}'s all batches".format(index, people_name))
        for model in LIST_MODEL:
            write_path = os.path.join(TEST_PATH, model + ".json")
            SavePicsAsJson(dict[model], write_path)
            PRINT_LOG.info("Write Test data Model {} Batches successfully".format(model))

    def WriteTrain1Batches():
        model_num_by_step = MODEL_NUMBER_BY_STEP
        start_index = 0
        while start_index < len(LIST_MODEL):
            dict = {}
            for model in LIST_MODEL:
                dict[model] = {}
            for index, people_name in enumerate(PEOPLE_TRAIN1):
                people_batch = GetPeopleBatchesDict(people_name)
                for model_index, model in enumerate(LIST_MODEL):
                    if model_index < start_index:
                        continue
                    if model_index >= start_index + model_num_by_step:
                        break
                    dict[model][people_name] = people_batch[model]
                PRINT_LOG.info("Get index{} people {}'s all batches".format(index, people_name))
            for model_index, model in enumerate(LIST_MODEL):
                if model_index < start_index:
                    continue
                if model_index > start_index + model_num_by_step:
                    break
                write_path = os.path.join(TRAIN_PATH1, model + ".json")
                SavePicsAsJson(dict[model], write_path)
                PRINT_LOG.info("Write Train1 data Model {} Batches successfully".format(model))
            start_index += model_num_by_step

    def WriteTrain2Batches():
        dict = {}
        for model in LIST_MODEL:
            dict[model] = {}
        for index, people_name in enumerate(PEOPLE_TRAIN2):
            people_batch = GetPeopleBatchesDict(people_name)
            for model_index, model in enumerate(LIST_MODEL):
                dict[model][people_name] = people_batch[model]
            PRINT_LOG.info("Get index{} people {}'s all batches".format(index, people_name))
        for model in LIST_MODEL:
            write_path = os.path.join(TRAIN_PATH2, model + ".json")
            SavePicsAsJson(dict[model], write_path)
            PRINT_LOG.info("Write Train2 Model {} Batches successfully".format(model))

    # WriteTestBatches()
    WriteTrain1Batches()
    # WriteTrain2Batches()








if __name__ == "__main__":
    pass

