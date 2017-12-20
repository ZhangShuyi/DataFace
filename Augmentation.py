import os
from InternalModule.Envs import *
from matplotlib import pyplot as plt
from InternalModule.LogSetting import PRINT_LOG, RECORD_FILE, ROOT_LOG
from InternalModule.FilePicAPI import *


class Data_Augmentation:
    def __init__(self, pic_name):
        self.pic_name = pic_name
        pass

    def load_origin_Img(self):
        path = os.path.join(PATH_DATA_NO_ALIGNED, self.pic_name)
        self.img = ReadPicture(path, color=PARA.COLOR, to_array=False)

    def get_augmentation(self):
        pass
