import pandas
from InternalModule.Envs import *
from InternalModule.FilePicAPI import *
from InternalModule.LogSetting import *
from CNN1 import *
import pickle
import threading


class FeatureID:
    @staticmethod
    def GetModelPartID(image_dict, model_name):
        TC = CNN1(input_size=[DATA_PARA_D_SIZE, DATA_PARA_D_SIZE, 1], label_size=DATA_PARA_PEOPLE_LIMIT_TRAIN1,
                  model_name=model_name,
                  feature_id_length=TRAIN_PARA_FEATURE_ID_LEN, train_keep_prob=0.5, device_num="/gpu:0")
        TC.build_model(False)
        TC.para_restore()
        TC.check_accuracy()
        partID_dict = {}
        for people_name in image_dict:
            partID_dict[people_name] = {}
            for pic_name in image_dict[people_name]:
                img = numpy.array(image_dict[people_name][pic_name]).reshape([DATA_PARA_D_SIZE, DATA_PARA_D_SIZE, 1])
                partID_dict[people_name][pic_name] = TC.get_feature(img).tolist()
        return partID_dict

    @staticmethod
    def WritePartID(people_list):
        operation_list = []
        for model_index, model_name in enumerate(LIST_MODEL):
            image_dict = {}
            for index, people_name in enumerate(people_list):
                path = os.path.join(PATH_DATA_BATCHES, model_name, str(people_name))
                with open(path, 'rb') as file:
                    image_dict[people_name] = pickle.load(file)
            operation_list.append(threading.Thread(target=FeatureID.Writeoperation, args=(model_name, image_dict)))
        PRINT_LOG.info("Thread operation was generated!")
        for operation in operation_list:
            operation.setDaemon(True)
            operation.start()
            operation.join()

    @staticmethod
    def Writeoperation(model_name, image_dict):
        save_path = os.path.join(PATH_DATA_FEATURE_ID, model_name)
        with open(save_path, 'wb') as file:
            pickle.dump(FeatureID.GetModelPartID(image_dict, model_name), file)
        PRINT_LOG.info("Model : {}  part ID was generated!".format(model_name))

    def __init__(self, flag):
        self.file_handler = []
        if flag == False:
            for model_name in LIST_MODEL:
                path = os.path.join(PATH_DATA_FEATURE_ID, model_name)
                with open(path, 'rb') as file:
                    self.file_handler.append(pickle.load(file))
        else:
            for model_name in LIST_MODEL:
                path = os.path.join(PATH_DATA_FEATURE_ID, model_name + '_t')
                with open(path, 'rb') as file:
                    self.file_handler.append(pickle.load(file))

    def GetAPicWholeID(self, identity,pic_name):
        feature_ID = []
        for file_handler in self.file_handler:
            feature_ID += list(file_handler[identity][pic_name])
        return feature_ID


if __name__ == "__main__":
    # WritePartID(LIST_TEST_PEOPLE)
    # WritePartID(LIST_TRAIN2_PEOPLE)
    FeatureID(False)
