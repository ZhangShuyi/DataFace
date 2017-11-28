# -*- coding: utf-8 -*-
"""
    Version 1.0
    Env
    all global variables
"""
import os
import random
import json
from InternalModule.LogSetting import ROOT_LOG, PRINT_LOG, RECORD_LOG

DEBUG = True

LANDMARK_SIZE = 68

D_SIZE = 32
K_SIZE = 16
TRAIN1_PRO = 0.6
TRAIN2_PRO = 0.2
MODEL_NUM = 77
FEATURE_ID_LENTH = 100
LAYER1_TEST_PRO = 0.2
# path
DATA_PATH = '/home/dcs/cas/CAS_Data/CAS_Origin'

TRAIN_PATH1 = '/home/dcs/cas/CAS_Data/CAS_Train1'
TRAIN_PATH2 = '/home/dcs/cas/CAS_Data/CAS_Train2'
TEST_PATH = '/home/dcs/cas/CAS_Data/CAS_Test'
TR_AND_TS_PATH = '/home/dcs/cas/CAS_Data/TR_TS.json'

LD_MODEL_PATH = '/media/dcs/Elements/DataFace/ModelAndTxt/shape_predictor_68_face_landmarks.dat'

LAYER1_MODEL = '/home/dcs/cas/CAS_Data/Layer1_Model'
LAYER2_MODEL = '/home/dcs/cas/CAS_Data/Layer2_Model'
FEATURE_PATH = '/home/dcs/cas/CAS_Data/FeatureID'
RESULT_PATH = '/home/dcs/cas/CAS_Data/Result'

BROAD_PATH = '/home/dcs/cas/CAS_Data/Broad'

#MODEL_LIST = ['0000']
MODEL_LIST = [ \
    '0000', '0028', '0070', \
    '0024', '0005', '0050', \
    '0051', '0008', \
    '0033', '0031', '0060', \
    '0047', '0057', \
    '0058', '0068', '0054' \
    ]

# list
PEOPLE_NAME_LIST = os.listdir(DATA_PATH)
PEOPLE_NUM = len(PEOPLE_NAME_LIST)
PEOPLE_TRAIN1 = []
PEOPLE_TRAIN2 = []
PEOPLE_TEST = []
if not os.path.exists(TR_AND_TS_PATH):
    for people_name in PEOPLE_NAME_LIST:
        r = random.random()
        if r < TRAIN1_PRO:
            PEOPLE_TRAIN1.append(people_name)
            continue
        if r < TRAIN1_PRO + TRAIN2_PRO:
            PEOPLE_TRAIN2.append(people_name)
        else:
            PEOPLE_TEST.append(people_name)
    data_dict = {}
    data_dict["Train1"] = PEOPLE_TRAIN1
    data_dict["Train2"] = PEOPLE_TRAIN2
    data_dict["Test"] = PEOPLE_TEST
    with open(TR_AND_TS_PATH, 'w') as file:
        json.dump(data_dict, file)
    ROOT_LOG.info(
        "TR and Ts was separated TR1:{} TR2:{} TS:{}".format(len(PEOPLE_TRAIN1), len(PEOPLE_TRAIN2), len(PEOPLE_TEST)))
else:
    with open(TR_AND_TS_PATH, 'r') as file:
        data_dict = json.load(file)
    PEOPLE_TRAIN1 = data_dict["Train1"]
    PEOPLE_TRAIN2 = data_dict["Train2"]
    PEOPLE_TEST = data_dict["Test"]
    ROOT_LOG.info(
        "TR and Ts was separated TR1:{} TR2:{} TS:{}".format(len(PEOPLE_TRAIN1), len(PEOPLE_TRAIN2), len(PEOPLE_TEST)))
