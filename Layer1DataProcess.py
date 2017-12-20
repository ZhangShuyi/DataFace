"""
    Layer1 Data Process
    version 1.0
"""
from InternalModule.Envs import *
import os
import pickle

class CrossValidationOperation:
    def __init__(self, cross_num, identity_list, name):
        self.cross_num = cross_num
        self.identity = identity_list
        self.name = name
        self.path = os.path.join(PATH_CROSS_VALIDATION, self.name)

    def generate_cross_validation(self, pic_info):
        self.dict = {}
        for index, identity in enumerate(self.identity):
            identity_info = pic_info[pic_info.identity == identity]
            identity_pic_num = len(identity_info)
            identity_cross_tag = [x % self.cross_num for x in range(identity_pic_num)]
            sub_index = 0
            for line_index, line_info in identity_info.iterrows():
                self.dict[line_info.pic_name] = identity_cross_tag[sub_index]
                sub_index += 1

    def save_cross_validation(self):
        with open(self.path, 'wb') as file:
            pickle.dump(self.dict, file)
        ROOT_LOG.info("Cross validation {} was saved at {}".format(self.name, self.path))

    def restore_cross_validation(self):
        with open(self.path, 'rb') as file:
            self.dict = pickle.load(file)

    def search_cross_validation(self, pic_name):
        return self.dict[pic_name]


if __name__ == "__main__":
    c = CrossValidationOperation(10, LIST_PEOPLE_LAYER1, "layer1_1000_10")
    # c.generate_cross_validation(FRAME_TRAIN1_INFO)
    # c.save_cross_validation()
    # c.restore_cross_validation()
    # print(c.dict)
    pass
