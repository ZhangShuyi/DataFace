import cv2
import dlib
import numpy as np
import matplotlib as plt
import matplotlib.pyplot
import math
import os

path = "D:\\data1\\data_train"
save_path = "D:\\data1\\data_train2"
predictor_path = 'D:\\face project\\dlib_face\\shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'D:\\face project\\dlib_face\\dlib_face_recognition_resnet_model_v1.dat'
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)



def face_alignment(faces, show=False):
    #print(faces.shape)
    if len(faces.shape) == 4 and faces.shape[3] == 1:
        faces = faces.reshape(
            faces.shape[:-1])  # if gray, turns to num * width * height, no channel axis 如果是灰度图，去掉最后一维，否则predictor会报错
    num = faces.shape[0]
    faces_aligned = np.zeros(shape=faces.shape, dtype=np.uint8)
    predictor = dlib.shape_predictor(predictor_path)  # 用来预测关键点
    for i in range(num):
        img = faces[i]
        rec = dlib.rectangle(0, 0, img.shape[0], img.shape[1])
        shape = predictor(np.uint8(img), rec)  # 注意输入的必须是uint8类型
        order = [36, 45, 30, 48, 54]  # left eye, right eye, nose, left mouth, right mouth  注意关键点的顺序，这个在网上可以找
        if show:
            plt.pyplot.figure()
            plt.pyplot.imshow(img, cmap='gray')
            for j in order:
                x = shape.part(j).x
                y = shape.part(j).y
                plt.pyplot.scatter(x, y)
        eye_center = ((shape.part(36).x + shape.part(45).x) * 1. / 2,
                      (shape.part(36).y + shape.part(45).y) * 1. / 2)
        dx = (shape.part(45).x - shape.part(36).x)
        dy = (shape.part(45).y - shape.part(36).y)
        angle = math.atan2(dy, dx) * 180. / math.pi
        RotateMatrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)  # 计算仿射矩阵
        RotImg = cv2.warpAffine(img, RotateMatrix, (img.shape[0], img.shape[1]))  # 进行放射变换，即旋转
        faces_aligned[i] = RotImg
    return faces_aligned


def align_face():
    for face_name in os.listdir(path):
        if not os.path.exists(os.path.join(save_path, face_name)):
            os.mkdir(os.path.join(save_path, face_name))
        for face_img in os.listdir(os.path.join(path, face_name)):
            img = cv2.imread(os.path.join(path, face_name, face_img))
            aligned = face_alignment(np.array([img]))
            cv2.imwrite(os.path.join(save_path, face_name, face_img), aligned[0])

        print("finised %s" % os.path.join(path, face_name))


if __name__ == "__main__":
    align_face()
