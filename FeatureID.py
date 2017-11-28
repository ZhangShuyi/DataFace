import pandas
from InternalModule.Envs import *
from InternalModule.FilePicAPI import *
from InternalModule.LogSetting import *
from CNN1 import *


def GetTrain2ForModel(model_name):
    return ReadTrain2Data(model_name)


def GetTestForModel(model_name):
    return ReadTestData(model_name)


def GetModelPartID(image_dict, model_name):
    C = CNN1(input_size=[32, 32, 1], label_size=len(PEOPLE_TRAIN1), model_name=model_name,
             feature_id_length=FEATURE_ID_LENTH)
    C.build_model(False)
    C.restore_para()
    partID_dict = {}
    for people_name in image_dict:
        partID_dict[people_name] = {}
        for pic_name in image_dict[people_name]:
            partID_dict[people_name][pic_name] = C.get_feature(image_dict[people_name][pic_name]).tolist()
    del C
    return partID_dict


# 2 times


def WriteTestPartID():
    for i, model_name in enumerate(MODEL_LIST):
        img_dict = GetTestForModel(model_name)
        partID_dict = GetModelPartID(img_dict, model_name)
        write_path = os.path.join(FEATURE_PATH, model_name + "_t.json")
        with open(write_path, 'w') as file:
            json.dump(partID_dict, file)
        PRINT_LOG.info("Model {}'s feature ID was writen".format(model_name))


def WriteTrain2PartID():
    for i, model_name in enumerate(MODEL_LIST):
        img_dict = GetTrain2ForModel(model_name)
        partID_dict = GetModelPartID(img_dict, model_name)
        write_path = os.path.join(FEATURE_PATH, model_name + ".json")
        with open(write_path, 'w') as file:
            json.dump(partID_dict, file)
        PRINT_LOG.info("Model {}'s feature ID was writen".format(model_name))


def GetAPicWholeID(pic_name, csv_handler_list):
    feature_ID = []
    for csv_handler in csv_handler_list:
        feature_ID += list(csv_handler[[pic_name]])
    return feature_ID


if __name__ == "__main__":
    WriteTestPartID()
    WriteTrain2PartID()
