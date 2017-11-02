import json
import os
import shutil

import Code.CNN_net as CNN_net
import Code.feature_data as feature_data
import cv2
import numpy

import InternalModule.ReadPicAPI as Read_API

PIC_PATH = "data3\datas_face_test\AbhishekBachan\P00100019.jpg"


def get_pic_batches(picture_path):
    json_data = feature_data.feature_one_picture(picture_path)
    return Read_API.picture_batches_with_json(picture_path, json_data, show=False)


def get_part_ID(batch, model_name):
    C = CNN_net.CNN1(input_height=32, input_width=32, label_size=59, input_channel=1, model_name=model_name)
    C.build_model(False)
    C.restore_para()
    return C.get_feature(batch)


def write_part_ID(path, model_name, write_path):
    C = CNN_net.CNN1(input_height=32, input_width=32, label_size=59, input_channel=1, model_name=model_name)
    C.build_model(False)
    C.restore_para()
    write_dict = {}
    for people_name in os.listdir(path):
        batches_path = os.path.join(path, people_name, model_name)
        for file_name in os.listdir(batches_path):
            tag = people_name + file_name.split('_')[1]
            img = cv2.imread(os.path.join(batches_path, file_name))
            img = numpy.array(img[:, :, :1], dtype=numpy.float32)
            cv2.normalize(img, img, alpha=1, beta=0, norm_type=cv2.NORM_INF)
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC).reshape(32, 32, 1)
            part_ID = C.get_feature(img)
            write_dict[tag] = part_ID.tolist()
    with open(write_path, 'w') as file_object:
        json.dump(write_dict, file_object)
    file_object.close()
    print("Info: write part ID {} completed".format(model_name))


def movefile(path, write_path):
    for model_name in os.listdir(path):
        for file_name in os.listdir(os.path.join(path, model_name)):
            shutil.copyfile(os.path.join(path, model_name, file_name), os.path.join(write_path, file_name))
    print("Info: move file {}".format(path))


if __name__ == "__main__":
    # write_feature_ID_file("D:\data1\data3\datas_face_test_feature\AbhishekBachan\\0000")
    WRITE_PATH = "D:\data1\data3\\featureID_test"
    PATH = "D:\data1\data3\datas_face_test_feature"
    '''
    for people_name in os.listdir(PATH):
        if not os.path.exists(os.path.join(WRITE_PATH, people_name)):
            os.mkdir(os.path.join(WRITE_PATH, people_name))
        movefile(os.path.join(PATH, people_name), os.path.join(WRITE_PATH, people_name))
    ''',

    for i, model_name in enumerate(os.listdir(os.path.join(PATH, os.listdir(PATH)[0]))):
        if i < 58:
            continue
        write_part_ID(PATH, model_name, os.path.join(WRITE_PATH, model_name + ".json"))
