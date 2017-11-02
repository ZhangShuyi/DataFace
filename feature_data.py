import os
import dlib
import json
import cv2
import Read_API

PATH = "D:\\data1\\data3\\datas_face_train"
JSON_PATH = "D:\\data1\\data3\\datas_face_train_json"
FEATURE_PATH = "D:\\data1\\data3\\datas_face_train_feature"

T_PATH = "D:\\data1\\data3\\datas_face_test"
T_JSON_PATH = "D:\\data1\\data3\\datas_face_test_json"
T_FEATURE_PATH = "D:\\data1\\data3\\datas_face_test_feature"

predictor_path = 'D:\\face project\\dlib_face\\shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'D:\\face project\\dlib_face\\dlib_face_recognition_resnet_model_v1.dat'
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)

feature_map = {}


def feature_one_picture(path):
    picture_diction = {}
    img = cv2.imread(path)
    rect = dlib.rectangle(0, 0, 64, 64)
    try:
        shape = sp(img, rect)
    except:
        print("No face!")
        return
    for i in range(68):
        picture_diction["%d" % i] = [shape.part(i).x, shape.part(i).y]
    return picture_diction


def create_people_json(people_name, pictures_path, write_path):
    people_diction = {}
    picture_list = os.listdir(pictures_path)
    for picture in picture_list:
        people_diction[picture] = feature_one_picture(os.path.join(pictures_path, picture))
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


if __name__ == "__main__":
    # write_with_feature("D:\\data1\\datas_face5_test", "D:\\data1\\datas_face6_test")

    for name in os.listdir(PATH):
        create_people_json(people_name=name, pictures_path=os.path.join(PATH, name), write_path=JSON_PATH)

    generate_batches(PATH, JSON_PATH, FEATURE_PATH)

    for name in os.listdir(T_PATH):
        create_people_json(people_name=name, pictures_path=os.path.join(T_PATH, name), write_path=T_JSON_PATH)

    generate_batches(T_PATH, T_JSON_PATH, T_FEATURE_PATH)
