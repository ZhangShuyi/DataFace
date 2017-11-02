import cv2
import os
import numpy

save_path = "D:\\data1\\datas_face3_test"
read_path = "D:\\data1\\datas_face2_test"


def filter_one_file(path):
    if not os.path.exists(os.path.join(save_path, path)):
        os.mkdir(os.path.join(save_path, path))
    for i, file in enumerate(os.listdir(os.path.join(read_path, path))):
        img = cv2.imread(os.path.join(read_path, path, file))
        # print(numpy.array(img).shape)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(os.path.join(save_path, path, file), gray)
    print("Finish %s" % os.path.join(read_path, path))


if __name__ == "__main__":
    for i, file_path in enumerate(os.listdir(read_path)):
        filter_one_file(file_path)
