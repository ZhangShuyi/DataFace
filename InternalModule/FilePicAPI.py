# -*- coding: utf-8 -*-
"""
    Version 1.0
    _ReadPicAPI
    Use for read Pic from data for input of CNN net
    Version 1.1
    fix some problem in Write function
"""
import os
import enum
import cv2
import numpy
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
    except FileNotFoundError:
        ROOT_LOG.error(" FileNotFoundError When Read image from {}".format(path))
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


def LoadFilePicInMemoryWithTag(path, d_size, tag=None, postfix='jpg', color=PARA.GRAY, normalization=PARA.NO_NORM):
    data = []
    dir_name_list = ScanFile(path, postfix=postfix)
    for dir_name in dir_name_list:
        data_line = []
        pic_path = os.path.join(path, dir_name)
        img = ReadPicture(pic_path, d_size=d_size, color=color, normalization=normalization)
        if tag is None:
            data.append(img)
        else:
            data_line.append(img)
            data_line.append(tag)
            data.append(data_line)
    ROOT_LOG.info("File {} image reading was success".format(path))
    return data


def LoadFilePicInMemoryWithAddress(path, postfix='jpg', color=PARA.GRAY):
    data = {}
    dir_name_list = ScanFile(path, postfix=postfix)
    for dir_name in dir_name_list:
        pic_path = os.path.join(path, dir_name)
        if color == PARA.GRAY:
            img = cv2.imread(pic_path, cv2.IMREAD_GRAYSCALE)
        elif color == PARA.COLOR:
            img = cv2.imread(pic_path, cv2.IMREAD_COLOR)
        data[pic_path] = img
    ROOT_LOG.info("File {} image reading was success".format(path))
    return data


if __name__ == "__main__":
    ReadPicture("D:\DataFromInternet\AbhishekBachan\\0001.jpg", d_size=10, normalization=PARA.MAX_32FLOAT,
                color=PARA.GRAY)
