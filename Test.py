"""
    Test.py
"""
import os
import pandas as pd
from InternalModule.FilePicAPI import ReadPicture, PARA
from CNN1 import CNN1


D_SIZE = 64
K_SIZE = 32

PIC_PATH = "/home/dcs/Desktop/Feature/0005/00000046/FM_000046_IEU+00_PD+00_EN_A0_D0_T0_BB_M0_R1_S0.tif.jpg"
FILE_PATH = "/home/dcs/Desktop/Feature/0076"
CSV_PATH = "/home/dcs/Desktop/CSV/0076.csv"
# landmark = GetPicLandmarkDict(img)
# batches = GetPicBatches(img, d_size=D_SIZE, k_size=K_SIZE, landmark_coordinate=landmark)

MODEL_LIST = ["04%d" % model_name for model_name in range(77)]
PEOPLE_SIZE = 1042
model_name = "0076"
C1 = CNN1(input_size=[32, 32, 1], label_size=PEOPLE_SIZE, model_name=model_name)
C1.build_model(False)
C1.restore_para()


def loadDataInMemory(path, csv_path):
    csv_data = pd.read_csv(csv_path).values
    ts_data = []
    ts_tag = []
    for index, line in enumerate(csv_data):
        [number, people_name, pic_name, train_tag] = line
        if train_tag == 0:
            pass
        else:
            lab = int(people_name)
            img = ReadPicture(os.path.join(path, "%08d" % people_name, pic_name), d_size=32, color=PARA.GRAY,
                              normalization=PARA.MAX_32FLOAT)
            ts_data.append(img)
            ts_tag.append(lab)
    return ts_data, ts_tag


data_l, tag_l = loadDataInMemory(FILE_PATH, CSV_PATH)
for i in range(len(data_l)):
    print(C1.get_identity(data_l[i]))
    print(tag_l[i])
