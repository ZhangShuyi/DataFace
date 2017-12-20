"""
    detect face
    version 2.0
    Show detail
"""
import cv2
import dlib
import sys
import dlib
from skimage import io
import os
import numpy
import pickle
from InternalModule.FilePicAPI import ReadPicture
from InternalModule.Envs import *
from matplotlib import pyplot as plt


class detectFace:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor(FILE_LD_MODEL)

    def GetPicFace(self, img):
        if img is None:
            return False
        faces_rect = self.face_detector(img, 1)
        if len(faces_rect) <= 0:
            PRINT_LOG.warn("GetPicFace have no face!")
            return False
        for k, d in enumerate(faces_rect):
            shape = self.landmark_predictor(img, d)
            for i in range(68):
                pt = shape.part(i)
                plt.plot(pt.x, pt.y, 'ro')
                plt.imshow(img)
                plt.show()

    def GetPicFaceV2(self, img):
        faces = self.detector(img, 1)
        face_img = []
        pic_shape = numpy.array(img).shape
        if (len(faces) > 0):
            for k, d in enumerate(faces):
                top = max(0, d.top() - 20)
                left = max(0, d.left() - 20)
                bottom = min(pic_shape[0], d.bottom() + 20)
                right = min(pic_shape[1], d.right() + 20)
                face_img.append(img[top:bottom, left:right])
        return face_img

    def ShowDetail(self, img):
        fig = plt.figure()
        face_list = self.GetPicFaceV2(img)
        for index, face in enumerate(face_list):
            ax = fig.add_subplot(1, len(face_list), index + 1)
            ax.imshow(face, cmap='gray')
            print(numpy.array(face.shape))
        fig.show()
        fig.waitforbuttonpress()

    def ReadAPickle(self, path):
        with open(path, 'rb') as file:
            data = pickle.load(file)
        print(type(data))

    # def FindAllFace(self, path, save_path, group_size=100):
    #     pic_dict = {}
    #     group_index = 1
    #     for index, pic_name in enumerate(os.listdir(path)):
    #         if (index + 1) % group_size == 0:
    #             with open(os.path.join(save_path, "%08d" % group_index), 'wb') as file:
    #                 pickle.dump(pic_dict, file)
    #             file.close()
    #             pic_dict = []
    #             PRINT_LOG.info("group index {} was generated".format(group_index))
    #             group_index += 1
    #         face = self.GetPicFaceV2(ReadPicture(os.path.join(path, pic_name), to_array=False))
    #         if len(face) == 0:
    #             PRINT_LOG.warn("index {} pic_name {} have no face!".format(index, pic_name))
    #             continue
    #         else:
    #             pic_dic


if __name__ == "__main__":
    # FindAllFace(ORIGIN_DATA_PATH, PICKLE_PATH)
    # ReadaPickle(os.path.join(PICKLE_PATH, os.listdir(PICKLE_PATH)[0]))
    pass