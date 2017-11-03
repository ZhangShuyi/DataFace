# -*- coding: utf-8 -*-
"""
    Version 1.0
    _ReadPicAPI
    Use for read Pic from data for input of CNN net
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
    MAX_64FLOAT = 6


def ReadPicture(path, d_size=0, color=PARA.GRAY, normalization=PARA.NO_NORM, show=False):
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
        img = cv2.resize(img, dsize=(d_size, d_size), interpolation=cv2.INTER_CUBIC)
    img = img.reshape(d_size, d_size, img_c)
    print(img.shape)
    return img


def scan_files(directory, prefix=None, postfix=None):
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


def load_all_pictures_in_file(path, tag=None, dst_size=None):
    data = []
    dir_name_list = scan_files(path)
    for dir_name in dir_name_list:
        data_line = []
        try:
            img = cv2.imread(dir_name)
            img = numpy.array(img[:, :, :1], dtype=numpy.float32)
            if dst_size:
                img2 = cv2.resize(img, dsize=(dst_size, dst_size), interpolation=cv2.INTER_AREA).reshape(dst_size,
                                                                                                         dst_size, 1)
            cv2.normalize(img2, img2, alpha=1, beta=0, norm_type=cv2.NORM_INF)
            data_line.append(img2)
            data_line.append(tag)
        except FileNotFoundError:
            print("File Not Found Error")
        except TypeError:
            # print("Type Error{}".format(dir_name))
            continue
        data.append(data_line)
    return data


def feature_map_with_json(path, json_data, ksize=10, show=False):
    img = cv2.imread(path)
    img = numpy.array(img, dtype=numpy.float32)
    cv2.normalize(img, img, alpha=1, beta=0, norm_type=cv2.NORM_INF)
    feature_map = []
    for i in range(68):
        i_s = "%d" % i
        left = max(0, json_data[i_s][0] - ksize)
        right = min(63, json_data[i_s][0] + ksize)
        top = max(0, json_data[i_s][1] - ksize)
        bottom = min(63, json_data[i_s][1] + ksize)
        feature_map.append(img[left:right, top:bottom, :])
        if show:
            plt.subplot(7, 10, i + 1)
            plt.axis('off')
            plt.imshow(feature_map[i])
    if show:
        plt.show()
    return feature_map


def load_all_batches_in_file_with_json(path, json_file):
    try:
        with open(os.path.join(path, json_file), 'r')as file_object:
            contents = json.load(file_object)
    except FileNotFoundError:
        print("Not Found json File")
        return
    data = []
    for pic_name in os.listdir(path):
        if pic_name.endswith('jpg'):
            data.append(picture_batches_with_json(os.path.join(path, pic_name), ksize=5, json_data=contents[pic_name],
                                                  show=True))
    return data


def picture_batches_with_json(path, json_data, show=False):
    img = cv2.imread(path)
    img = numpy.array(img, dtype=numpy.float32)
    img = numpy.array(img[:, :, :1], dtype=numpy.float32)
    # cv2.normalize(img, img, alpha=1, beta=0, norm_type=cv2.NORM_INF)
    batches = []
    size = img.shape
    batches.append(img.reshape(64, 64, 1))
    batches.append(
        cv2.resize(img[:int(size[0] / 2), :, :], dsize=(64, 64), interpolation=cv2.INTER_CUBIC).reshape(64, 64, 1))
    batches.append(
        cv2.resize(img[:int(size[0] / 2), :, :], dsize=(64, 64), interpolation=cv2.INTER_CUBIC).reshape(64, 64, 1))
    batches.append(
        cv2.resize(img[:, :int(size[1] / 2), :], dsize=(64, 64), interpolation=cv2.INTER_CUBIC).reshape(64, 64, 1))
    batches.append(
        cv2.resize(img[:, int(size[1] / 2):, :], dsize=(64, 64), interpolation=cv2.INTER_CUBIC).reshape(64, 64, 1))
    # 32*32 1/4 4
    batches.append(
        cv2.resize(img[:int(size[0] / 2), :int(size[1] / 2), :], dsize=(64, 64), interpolation=cv2.INTER_CUBIC).reshape(
            64, 64, 1))
    batches.append(
        cv2.resize(img[:int(size[0] / 2), int(size[1] / 2):, :], dsize=(64, 64), interpolation=cv2.INTER_CUBIC).reshape(
            64, 64, 1))
    batches.append(
        cv2.resize(img[int(size[0] / 2):, :int(size[1] / 2), :], dsize=(64, 64), interpolation=cv2.INTER_CUBIC).reshape(
            64, 64, 1))
    batches.append(
        cv2.resize(img[int(size[0] / 2):, int(size[1] / 2):, :], dsize=(64, 64), interpolation=cv2.INTER_CUBIC).reshape(
            64, 64, 1))

    # feature map
    ksize = 10
    for i, coordinate in enumerate(json_data):
        i_s = "%d" % i
        left = max(0, json_data[i_s][0] - ksize)
        right = min(63, json_data[i_s][0] + ksize)
        top = max(0, json_data[i_s][1] - ksize)
        bottom = min(63, json_data[i_s][1] + ksize)
        try:
            batches.append(
                cv2.resize(img[left:right, top:bottom, :], dsize=(64, 64), interpolation=cv2.INTER_CUBIC).reshape(64,
                                                                                                                  64,
                                                                                                                  1))
        except cv2.error:
            print("out picture: left {}, right{}, top{} ,bottom{}".format(left, right, top, bottom))
            batches.append(None)
    if show:
        for i, pic in enumerate(batches):
            plt.subplot(10, 10, i + 1)
            plt.axis('off')
            plt.imshow(batches[i].reshape(64, 64))
    plt.show()
    return batches


def load_all_batches_in_file(path, json_file):
    with open(json_file, 'r')as file_object:
        contents = json.load(file_object)
    data = []
    for pic in os.listdir(path):
        if pic.endswith('jpg'):
            data.append(picture_batches_with_json(os.path.join(path, pic), json_data=contents[pic], show=False))
    return data


def load_random_data_with_batches(path, json_file):
    with open(os.path.join(path, json_file), 'r')as file_object:
        contents = json.load(file_object)
    for pic in os.listdir(path):
        if pic.endswith('jpg'):
            return picture_batches_with_json(os.path.join(path, pic), json_data=contents[pic], show=False)
        else:
            continue


if __name__ == "__main__":
    ReadPicture("D:\DataFromInternet\AbhishekBachan\\0001.jpg", d_size=10, color=PARA.GRAY)
