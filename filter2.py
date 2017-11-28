import cv2
import dlib
import os
import numpy

sava_path = "D:\\data1\\datas_face2"
path = "D:\\data1\\datas_face"
predictor_path = 'D:\\face project\\dlib_face\\shape_predictor_68_face_landmarks.dat'
face_rec_model_path = 'D:\\face project\\dlib_face\\dlib_face_recognition_resnet_model_v1.dat'
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
facerec = dlib.face_recognition_model_v1(face_rec_model_path)
thresold = 0.6


def get_directory(path):
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return directories


def get_filepath(path, directory):
    filenames = [d for d in os.listdir(os.path.join(path, directory))]
    return filenames


def filter_one_file(path, file_name, file_num=0):
    des = []
    positive_sample = 1
    negative_sample = 1

    if os.path.exists(os.path.join(path, '_.jpg')):
        print("have tag!!!")
        img = cv2.imread(os.path.join(path, '_.jpg'))
        rect = dlib.rectangle(0, 0, 64, 64)
        shape = sp(img, rect)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        des = face_descriptor
        try:
            os.mkdir(os.path.join(sava_path, file_name))
            cv2.imwrite(os.path.join(sava_path, file_name, "P%03d%05d.jpg" % (file_num, positive_sample)), img)
            os.rename(os.path.join(path, '_.jpg'), os.path.join(path, "P%03d%05dv1.jpg" % (file_num, positive_sample)))
            positive_sample += 1
        except FileExistsError:
            print("already did")

    for i, file in enumerate(os.listdir(path)):
        # print(os.path.join(path, file))
        img = cv2.imread(os.path.join(path, file))
        rect = dlib.rectangle(0, 0, 64, 64)
        shape = sp(img, rect)
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        if i == 0 and des == []:
            des = face_descriptor
            try:
                os.mkdir(os.path.join(sava_path, file_name))
                cv2.imwrite(os.path.join(sava_path, file_name, "P%03d%05d.jpg" % (file_num, positive_sample)), img)
                os.rename(os.path.join(path, file), os.path.join(path, "P%03d%05dv1.jpg" % (file_num, positive_sample)))
                positive_sample += 1
            except FileExistsError:
                print("already did")

        else:
            dist = numpy.linalg.norm(numpy.array(des) - numpy.array(face_descriptor))
            # print(dist)
            if dist < thresold:
                cv2.imwrite(os.path.join(sava_path, file_name, "P%03d%05d.jpg" % (file_num, positive_sample)), img)
                os.rename(os.path.join(path, file), os.path.join(path, "P%03d%05dv1.jpg" % (file_num, positive_sample)))
                positive_sample += 1
            else:
                os.rename(os.path.join(path, file), os.path.join(path, "ZN%03d%05dv1.jpg" % (file_num, negative_sample)))
                negative_sample += 1
                # dlib.hit_enter_to_continue()


def change_name(path, j):
    for file_path in os.listdir(path):
        # print(os.path.join(path, "%03d" % j + file_path))
        os.rename(os.path.join(path, file_path), os.path.join(path, "%03d" % j + file_path))


if __name__ == "__main__":
    for i, file_path in enumerate(os.listdir(path)):
        # change_name(os.path.join(path, file_path), i)
        filter_one_file(os.path.join(path, file_path), file_path, i + 1)
        print("finish %s" % os.path.join(path, file_path))
