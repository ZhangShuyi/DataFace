"""
    PicToJson
    version 1.0
    Version 2.0
"""
import json
import cv2
import pickle
from InternalModule.FilePicAPI import *
from InternalModule.LogSetting import *
from InternalModule.Envs import *
from matplotlib import pyplot as plt


class JsonOperation:
    def PicToJson(self, pic_info):
        data = {}
        for index, row_info in pic_info.iterrows():
            identity = row_info.identity
            pic_name = row_info.pic_name
            img = ReadPicture(os.path.join(PATH_DATA_ALIGNED, pic_name), DATA_PARA_D_SIZE).tolist()
            if identity not in data:
                data[identity] = {}
            data[identity][pic_name] = img
        for identity in data:
            path = os.path.join(PATH_JSON_STYLE, str(identity) + ".json")
            if os.path.exists(path):
                with open(path, 'r') as file:
                    origin_data = json.load(file)
                file.close()
                new_data = dict(origin_data, **data[identity])
                with open(path, 'w') as file:
                    json.dump(new_data, file)

            else:
                with open(path, 'w') as file:
                    json.dump(data[identity], file)

    def PicToJsonAllProcess(self, each_step=10000):
        line_number = FRAME_TRAIN1_INFO.shape[0]
        PRINT_LOG.info("Train layer total sample {}".format(line_number))
        start = 0
        index = 0
        while True:
            index += 1
            end = min(start + each_step, line_number)
            pic_info = FRAME_TRAIN1_INFO[start:end]
            self.PicToJson(pic_info)
            PRINT_LOG.info("Index:{} (start {} / end {}) Pic to Json finished".format(index, start, end))
            if end == line_number:
                PRINT_LOG.info("All Pic to Json was finished!")
                break
            start = end

    def ShowAJson(self, json_path):
        fig = plt.figure()
        img_list = []
        with open(json_path, 'r') as file:
            data = json.load(file)
        for pic_name in data:
            img_list.append(numpy.array(data[pic_name]).reshape(DATA_PARA_D_SIZE, DATA_PARA_D_SIZE, 1))
        total = len(img_list)
        width = 5
        height = total / 5 + 1
        for index, img in enumerate(img_list):
            img = img[:, :, 0]
            ax = fig.add_subplot(width, height, index + 1)
            ax.imshow(img, cmap='gray')
        fig.show()
        fig.waitforbuttonpress()

    def PicToJsonV2(self, pic_info):
        data = {}
        for index, row_info in pic_info.iterrows():
            identity = row_info.identity
            pic_name = row_info.pic_name
            img = ReadPicture(os.path.join(PATH_DATA_ALIGNED, pic_name), DATA_PARA_D_SIZE).tolist()
            if identity not in data:
                data[identity] = {}
            data[identity][pic_name] = img
        for identity in data:
            path = os.path.join(PATH_JSON_STYLE, str(identity) + ".json")
            if os.path.exists(path):
                with open(path, 'r') as file:
                    origin_data = json.load(file)
                file.close()
                new_data = dict(origin_data, **data[identity])
                with open(path, 'w') as file:
                    json.dump(new_data, file)
            else:
                with open(path, 'w') as file:
                    json.dump(data[identity], file)


class PickleOperation(object):
    def __init__(self, D_size, save_path):
        self.D_size = DATA_PARA_D_SIZE
        self.save_path = save_path

    def ShowAPickle(self, people_name):
        fig = plt.figure()
        img_list = []
        pickle_path = os.path.join(self.save_path, people_name)
        with open(pickle_path, 'rb') as file:
            data = pickle.load(file)
        for pic_name in data:
            img_list.append(numpy.array(data[pic_name]).reshape(self.D_size, self.D_size, 1))
        total = len(img_list)
        width = 5
        height = total / 5 + 1
        for index, img in enumerate(img_list):
            img = img[:, :, 0]
            ax = fig.add_subplot(width, height, index + 1)
            ax.imshow(img, cmap='gray')
        fig.show()
        fig.waitforbuttonpress()

    def PicToPickle(self, pic_info):
        data = {}
        for index, row_info in pic_info.iterrows():
            identity = row_info.identity
            pic_name = row_info.pic_name
            img = ReadPicture(os.path.join(PATH_DATA_ALIGNED, pic_name), d_size=self.D_size).tolist()
            if identity not in data:
                data[identity] = {}
            data[identity][pic_name] = img
        for identity in data:
            path = os.path.join(self.save_path, str(identity))
            if os.path.exists(path):
                with open(path, 'rb') as file:
                    origin_data = pickle.load(file)
                file.close()
                new_data = dict(origin_data, **data[identity])
                with open(path, 'wb') as file:
                    pickle.dump(new_data, file)
            else:
                with open(path, 'wb') as file:
                    pickle.dump(data[identity], file)

    def PicToPickleAllProcess(self, each_step=10000):
        line_number = FRAME_TRAIN1_INFO.shape[0]
        PRINT_LOG.info("Train layer total sample {}".format(line_number))
        start = 0
        index = 0
        while True:
            index += 1
            end = min(start + each_step, line_number)
            pic_info = FRAME_TRAIN1_INFO[start:end]
            self.PicToPickle(pic_info)
            PRINT_LOG.info("Index:{} (start {} / end {}) Pic to Pickle finished".format(index, start, end))
            if end == line_number:
                PRINT_LOG.info("All Pic to Pickle was finished!")
                break
            start = end

    def ReadAPickle(self, people_name):
        path = os.path.join(self.save_path, str(people_name))
        with open(path, 'rb') as file:
            data = pickle.load(file)
        return data


if __name__ == "__main__":
    total = 0
    for people_name in os.listdir(PATH_PICKLE_STYLE):
        #     total += len(PickleOperation(D_size=D_SIZE, save_path=PICKLE_PATH).ReadAPickle(people_name))
        # print(total)
        PickleOperation(D_size=DATA_PARA_D_SIZE, save_path=PATH_PICKLE_STYLE).ShowAPickle(people_name)
        print(len(PickleOperation(D_size=DATA_PARA_D_SIZE, save_path=PATH_PICKLE_STYLE).ReadAPickle(people_name)))
