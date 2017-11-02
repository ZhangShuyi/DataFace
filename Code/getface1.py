import sys
import os
import dlib
from skimage import transform
import cv2
from skimage import io
from scipy.misc import imread

detector = dlib.get_frontal_face_detector()
# path\directory\file_name
path = "D:\\data1\\datas"
save_path = "D:\\data1\\datas_face"


def get_directory(path):
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return directories


def get_filepath(path, directory):
    filenames = [d for d in os.listdir(os.path.join(path, directory))]
    return filenames


directories = get_directory(path)

for d in directories:
    filenames = get_filepath(path, d)
    for f in filenames:
        file_path = os.path.join(path, d, f)
        print("Processing file: {}".format(file_path))
        try:
            img = cv2.imread(file_path)
        except:
            print("Invalid picture")
            continue

        try:

            for i, pos in enumerate(dets):
                face = img[pos.top():pos.bottom(), pos.left():pos.right()]
                face_resize = cv2.resize(face, (64, 64))
                if not os.path.exists(os.path.join(save_path, d)):
                    os.mkdir(os.path.join(save_path, d))

                print(os.path.join(save_path, d, f))
                cv2.imwrite(os.path.join(save_path, d, f), face_resize)
                # cv2.imshow('face', img)
                # cv2.imshow('face_resize', face_resize)
                # cv2.waitKey(0)
        except:
            print("Can't write")
