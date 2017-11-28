import dlib
import os
import numpy
from InternalModule.FilePicAPI import ReadPicture
LANDMARK_MODEL = 'ModelAndTxt\shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'D:\\face project\\dlib_face\\dlib_face_recognition_resnet_model_v1.dat'
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(LANDMARK_MODEL)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
THRESHOLD = 0.6


def filter_one_file(path):
    des = []
    positive_sample = 1
    negative_sample = 1
    # print(os.path.join(path, file))
    img = ReadPicture(path)
    rect = dlib.rectangle(0, 0, 64, 64)
    shape = sp(img, rect)
    face_descriptor = facerec.compute_face_descriptor(img, shape)
    if i == 0:
        des = face_descriptor
        try:
            os.rename(os.path.join(path, file), os.path.join(path, "P%08d.jpg" % positive_sample))
            positive_sample += 1
        except FileExistsError:
            print("already did")
    else:
        dist = numpy.linalg.norm(numpy.array(des) - numpy.array(face_descriptor))
        # print(dist)
        if dist < THRESHOLD:
            os.rename(os.path.join(path, file), os.path.join(path, "P%08d.jpg" % positive_sample))
            positive_sample += 1
        else:
            os.rename(os.path.join(path, file), os.path.join(path, "N%08d.jpg" % negative_sample))
            negative_sample += 1
            # dlib.hit_enter_to_continue()


def change_name(path, j):
    for file_path in os.listdir(path):
        # print(os.path.join(path, "%03d" % j + file_path))
        os.rename(os.path.join(path, file_path), os.path.join(path, "%03d" % j + file_path))


for i, file_path in enumerate(os.listdir(path)):
    change_name(os.path.join(path, file_path), i)
    # filter_one_file(os.path.join(path, file_path))
    print("finish %s" % os.path.join(path, file_path))
