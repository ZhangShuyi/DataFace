# -*- coding: utf-8 -*-
"""
    Version 1.0
    _ReadPicAPI
    Use for read Pic from data for input of CNN net
    Version 1.1
    fix some problem in Write function
"""
from InternalModule.Envs import *
import json
import os
import enum
import cv2
import numpy
import pandas as pd
from InternalModule.LogSetting import ROOT_LOG, RECORD_LOG, PRINT_LOG


class PARA(enum.Enum):
    GRAY = 1
    COLOR = 2
    NO_NORM = 3
    MAX_255INT = 4
    MAX_32FLOAT = 5


def ReadPicture(path, d_size=0, color=PARA.GRAY, normalization=PARA.NO_NORM, to_array=True, show=False,
                success_log=False):
    if to_array is False:
        if color == PARA.GRAY:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        elif color == PARA.COLOR:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
        return img
    try:
        if color == PARA.GRAY:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = numpy.array(img)
            img_h = img.shape[0]
            img_w = img.shape[1]
            img_c = 1
        elif color == PARA.COLOR:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            img_h = img.shape[0]
            img_w = img.shape[1]
            img_c = img.shape[2]
    except IndexError:
        PRINT_LOG.error("IndexError")
        return
    except FileNotFoundError:
        ROOT_LOG.error(" FileNotFoundError When Read image from {}".format(path))
        return
    if d_size is not 0:
        img_w = d_size
        img_h = d_size
    img = cv2.resize(img, dsize=(img_h, img_w), interpolation=cv2.INTER_CUBIC)
    img = numpy.array(img)
    img = img.reshape(img_h, img_w, img_c)
    if normalization == PARA.NO_NORM:
        img = img
    elif normalization == PARA.MAX_255INT:
        img = img.astype(numpy.int32)
        cv2.normalize(img, img, alpha=255, beta=0, norm_type=cv2.NORM_MINMAXM)
    elif normalization == PARA.MAX_32FLOAT:
        img = img.astype(numpy.float32)
        cv2.normalize(img, img, alpha=1.0, beta=0.0, norm_type=cv2.NORM_MINMAX)
    if show:
        cv2.imshow("pic", img)
        cv2.waitKey()
    if success_log:
        ROOT_LOG.info("Read image from {} success".format(path))
    return img


def WritePicture(img, save_path, show=False, success_log=False):
    try:
        cv2.imwrite(save_path, img)
    except FileNotFoundError:
        ROOT_LOG.error("FileNotFoundError When Write image from {}".format(save_path))

    if show:
        cv2.imshow("pic", img)
        cv2.waitKey()
    if success_log:
        ROOT_LOG.info("Write image from {} success".format(save_path))
    return


def LoadPicsAsDict(pic_path, d_size=0, color=PARA.GRAY, normalization=PARA.NO_NORM):
    dict = {}
    for pic_name in os.listdir(pic_path):
        dict[pic_name] = ReadPicture(
            os.path.join(pic_path, pic_name), d_size=d_size, color=color, normalization=normalization, to_array=False)
    # ROOT_LOG.info("LoadPicAsDict: path {} d_size {} total{} success".format(pic_path, d_size, len(dict)))
    return dict


def SavePicsAsJson(img_dict, json_path):
    with open(json_path, 'w') as file:
        json.dump(img_dict, file)
    ROOT_LOG.info("SavePicsAsJson: save_path {} success".format(json_path))


def LoadPicsFromJson(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
    for people_name in data:
        for pic_name in data[people_name]:
            data[people_name][pic_name] = numpy.array(data[people_name][pic_name]).reshape(D_SIZE, D_SIZE, 1)
    return data


def ScanFile(directory, prefix=None, postfix=None):
    files_list = []
    for root, sub_dirs, files in os.walk(directory):
        for special_file in files:
            if postfix:
                if special_file.endswith(postfix):
                    files_list.append(os.path.join(root, special_file))
            elif prefix:
                if special_file.startswith(prefix):
                    files_list.append(os.path.join(root, special_file))
            else:
                files_list.append(os.path.join(root, special_file))
    return files_list


def ReadTrain1Data(model_name):
    path = os.path.join(TRAIN_PATH1, model_name + ".json")
    data = LoadPicsFromJson(path)
    PRINT_LOG.info("Read Model {} Train 1 Data".format(model_name))
    return data


def ReadTrain2Data(model_name):
    path = os.path.join(TRAIN_PATH2, model_name + ".json")
    data = LoadPicsFromJson(path)
    PRINT_LOG.info("Read Model {} Train 2 Data".format(model_name))
    return data


def ReadTestData(model_name):
    path = os.path.join(TEST_PATH, model_name + ".json")
    data = LoadPicsFromJson(path)
    PRINT_LOG.info("Read Model {} Test  Data".format(model_name))
    return data


if __name__ == "__main__":
    pass
