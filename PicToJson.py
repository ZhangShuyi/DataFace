"""
    PicToJson
    version 1.0
"""
import json
from InternalModule.FilePicAPI import *
from InternalModule.LogSetting import *
from InternalModule.Envs import *


def PicToJson(pic_info):
    data = {}
    for index, row_info in pic_info.iterrows():
        identity = row_info.identity
        pic_name = row_info.pic_name
        img = ReadPicture(os.path.join(DATA_PATH, pic_name), D_SIZE).tolist()
        if identity not in data:
            data[identity] = {}
        data[identity][pic_name] = img
    for identity in data:
        path = os.path.join(JSON_PATH, str(identity) + ".json")
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


if __name__ == "__main__":
    line_number = TRAIN1_INFO.shape[0]
    PRINT_LOG.info("Train layer total sample {}".format(line_number))
    each_step = 10000
    start = 0
    index = 0
    while True:
        index += 1
        end = min(start + each_step, line_number)
        pic_info = TRAIN1_INFO[start:end]
        PicToJson(pic_info)
        PRINT_LOG.info("index {} Pic to Json finished".format(index))
        if end == line_number:
            PRINT_LOG.info("pic to json was finished!")
            break
        start = end
