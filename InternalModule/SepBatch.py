import os
import dlib
import json
import cv2
import matplotlib as plt
import InternalModule.FilePicAPI
from InternalModule.LogSetting *

PATH = "D:\\data1\\data3\\datas_face_train"
JSON_PATH = "D:\\data1\\data3\\datas_face_train_json"
FEATURE_PATH = "D:\\data1\\data3\\datas_face_train_feature"

T_PATH = "D:\\data1\\data3\\datas_face_test"
T_JSON_PATH = "D:\\data1\\data3\\datas_face_test_json"
T_FEATURE_PATH = "D:\\data1\\data3\\datas_face_test_feature"

LD_MODEL_PATH = '../ModelAndTxt/shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'D:\\face project\\dlib_face\\dlib_face_recognition_resnet_model_v1.dat'
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(LD_MODEL_PATH)
LANDMARK_SIZE = 68

feature_map = {}


def GetPicLandmark(img, img_name):
    picture_diction = {}
    rect = dlib.rectangle(0, 0, img.shape[0], img.shape[1])
    try:
        shape = sp(img, rect)
        for i in range(LANDMARK_SIZE):
            picture_diction["%d" % i] = [shape.part(i).x, shape.part(i).y]
    except KeyError:
        ROOT_LOG.warning("Img {} have no face!".format(img_name))
        return None
    return picture_diction


def create_people_json(people_name, pictures_path, write_path):
    people_diction = {}
    picture_list = os.listdir(pictures_path)
    for picture in picture_list:
        people_diction[picture] = GetPicLandmark(os.path.join(pictures_path, picture))
    json_path = os.path.join(write_path, people_name + ".json")
    with open(json_path, 'w') as file_object:
        json.dump(people_diction, file_object)
    print("generate json {}".format(pictures_path))


def generate_batches(path, json_path, write_path):
    for people_index, people_name in enumerate(os.listdir(path)):
        print("generate batches {}".format(people_name))
        if not os.path.exists(os.path.join(write_path, people_name)):
            os.mkdir(os.path.join(write_path, people_name))
        data = Read_API.load_all_batches_in_file(os.path.join(path, people_name),
                                                 os.path.join(json_path, people_name + ".json"))
        if not os.path.exists(os.path.join(write_path, people_name)):
            os.mkdir(os.path.join(write_path, people_name))
        for picture_index, picture in enumerate(data):
            for feature_index, feature in enumerate(picture):
                if not os.path.exists(os.path.join(write_path, people_name, "%04d" % feature_index)):
                    os.mkdir(os.path.join(write_path, people_name, "%04d" % feature_index))
                if feature is not None:
                    cv2.imwrite(os.path.join(write_path, people_name, "%04d" % feature_index,
                                             "f%04d_p%04d.jpg" % (feature_index, picture_index)),
                                feature.reshape(64, 64))
                else:
                    print("exist None: people_index {} feature_index {}, picture_index {}".format(people_index,
                                                                                                  feature_index,
                                                                                                  picture_index))


def ExtractFeatureWithJson(img, json_data, feature_size=10, show=False):
    h_min = 0
    w_min = 0
    h_max = img.shape[0]
    w_max = img.shape[1]
    feature_map = []
    for i, feature_index in enumerate(json_data):
        left = max(w_min, json_data[feature_index][1] - feature_size)
        right = min(w_max, json_data[feature_index][1] + feature_size)
        top = max(h_min, json_data[feature_index][0] - feature_size)
        bottom = min(h_max, json_data[feature_index][0] + feature_size)
        feature_map.append(img[top:bottom, left:right, :])
        if show:
            plt.subplot(7, 10, i + 1)
            plt.axis('off')
            plt.imshow(feature_map[i])
    if show:
        plt.show()
    return feature_map


if __name__ == "__main__":
    # write_with_feature("D:\\data1\\datas_face5_test", "D:\\data1\\datas_face6_test")

    for name in os.listdir(PATH):
        create_people_json(people_name=name, pictures_path=os.path.join(PATH, name), write_path=JSON_PATH)

    generate_batches(PATH, JSON_PATH, FEATURE_PATH)

    for name in os.listdir(T_PATH):
        create_people_json(people_name=name, pictures_path=os.path.join(T_PATH, name), write_path=T_JSON_PATH)

    generate_batches(T_PATH, T_JSON_PATH, T_FEATURE_PATH)
