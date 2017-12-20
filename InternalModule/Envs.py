# -*- coding: utf-8 -*-
"""
    Version 1.0
    Env
    all global variables
"""
import pandas
from InternalModule.LogSetting import ROOT_LOG, PRINT_LOG, RECORD_LOG

FLAG_DEBUG = True

# LANDMARK_SIZE = 68
#
# D_SIZE = 32
# K_SIZE = 16
# TRAIN1_PRO = 0.6
# TRAIN2_PRO = 0.2
# MODEL_NUM = 77
# FEATURE_ID_LENTH = 100
# LAYER1_TEST_PRO = 0.2
# # path
# DATA_PATH = '/home/dcs/cas/CAS_Data/CAS_Origin'
#
# TRAIN_PATH1 = '/home/dcs/cas/CAS_Data/CAS_Train1'
# TRAIN_PATH2 = '/home/dcs/cas/CAS_Data/CAS_Train2'
# TEST_PATH = '/home/dcs/cas/CAS_Data/CAS_Test'
# TR_AND_TS_PATH = '/home/dcs/cas/CAS_Data/TR_TS.json'
#
# LD_MODEL_PATH = '/media/dcs/Elements/shape_predictor_68_face_landmarks.dat'
#
# LAYER1_MODEL = '/home/dcs/cas/CAS_Data/Layer1_Model'
# LAYER2_MODEL = '/home/dcs/cas/CAS_Data/Layer2_Model'
# FEATURE_PATH = '/home/dcs/cas/CAS_Data/FeatureID'
# RESULT_PATH = '/home/dcs/cas/CAS_Data/Result'
#
# BROAD_PATH = '/home/dcs/cas/CAS_Data/Broad'
#
# # MODEL_LIST = ['0000']
# # MODEL_LIST = [ \
# #     '0000', '0028', '0070', \
# #     '0024', '0005', '0050', \
# #     '0051', '0008', \
# #     '0033', '0031', '0060', \
# #     '0047', '0057', \
# #     '0058', '0068', '0054' \
# #     ]
# MODEL_LIST = ['%04d' % i for i in range(MODEL_NUM)]
#
# # list
# PEOPLE_NAME_LIST = os.listdir(DATA_PATH)
# PEOPLE_NUM = len(PEOPLE_NAME_LIST)
# PEOPLE_TRAIN1 = []
# PEOPLE_TRAIN2 = []
# PEOPLE_TEST = []
# if not os.path.exists(TR_AND_TS_PATH):
#     for people_name in PEOPLE_NAME_LIST:
#         r = random.random()
#         if r < TRAIN1_PRO:
#             PEOPLE_TRAIN1.append(people_name)
#             continue
#         if r < TRAIN1_PRO + TRAIN2_PRO:
#             PEOPLE_TRAIN2.append(people_name)
#         else:
#             PEOPLE_TEST.append(people_name)
#     data_dict = {}
#     data_dict["Train1"] = PEOPLE_TRAIN1
#     data_dict["Train2"] = PEOPLE_TRAIN2
#     data_dict["Test"] = PEOPLE_TEST
#     with open(TR_AND_TS_PATH, 'w') as file:
#         json.dump(data_dict, file)
#     ROOT_LOG.info(
#         "TR and Ts was separated TR1:{} TR2:{} TS:{}".format(len(PEOPLE_TRAIN1), len(PEOPLE_TRAIN2), len(PEOPLE_TEST)))
# else:
#     with open(TR_AND_TS_PATH, 'r') as file:
#         data_dict = json.load(file)
#     PEOPLE_TRAIN1 = data_dict["Train1"]
#     PEOPLE_TRAIN2 = data_dict["Train2"]
#     PEOPLE_TEST = data_dict["Test"]
#     ROOT_LOG.info(
#         "TR and Ts was separated TR1:{} TR2:{} TS:{}".format(len(PEOPLE_TRAIN1), len(PEOPLE_TRAIN2), len(PEOPLE_TEST)))

TRAIN_PARA_WN_LAMBDA = 2
TRAIN_PARA_FEATURE_ID_LEN = 160

DATA_PARA_D_SIZE = 96
DATA_PARA_K_SIZE = 16
DATA_PARA_LAYER1_TEST_PRO = 0.05
# path
PATH_DATA_ALIGNED = '/media/dcs/TRANSCEND/CelebA/Img/img_align_celeba/img_align_celeba'
PATH_DATA_NO_ALIGNED = '/media/dcs/TRANSCEND/CelebA/Img/img_celeba.7z/data/img_celeba'
PATH_PICKLE_STYLE = '/media/dcs/TRANSCEND/CelebA/Img_binary'
PATH_JSON_STYLE = '/media/dcs/TRANSCEND/CelebA/Json'
PATH_CROSS_VALIDATION = '/media/dcs/TRANSCEND/CelebA/Cross'

FILE_TR_AND_TS = '/media/dcs/TRANSCEND/CelebA/Eval/list_eval_partition.txt'
FILE_IDENTITY = '/media/dcs/TRANSCEND/CelebA/Eval/identity_CelebA.txt'
FILE_LANDMARK = '/media/dcs/TRANSCEND/CelebA/Anno/list_landmarks_align_celeba.txt'
FILE_LD_MODEL = '/media/dcs/Elements/shape_predictor_68_face_landmarks.dat'

PATH_LAYER1_MODEL = '/media/dcs/TRANSCEND/CelebA/Layer1Model'
PATH_LAYER2_MODEL = '/media/dcs/TRANSCEND/CelebA/Layer2Model'

PATH_RESULT = '/media/dcs/TRANSCEND/CelebA/Result'

PATH_BROAD = '/media/dcs/TRANSCEND/CelebA/Broad'

# MODEL_LIST = ['0000']
# MODEL_LIST = [ \
#     '0000', '0028', '0070', \
#     '0024', '0005', '0050', \
#     '0051', '0008', \
#     '0033', '0031', '0060', \
#     '0047', '0057', \
#     '0058', '0068', '0054' \
#     ]
LIST_MODEL = ['0000']

# list
FRAME_TR_TS = pandas.DataFrame(pandas.read_csv(FILE_TR_AND_TS, delim_whitespace=True))
FRAME_IDENTITY = pandas.DataFrame(pandas.read_csv(FILE_IDENTITY, delim_whitespace=True))
FRAME_LANDMARK = pandas.DataFrame(pandas.read_csv(FILE_LANDMARK, delim_whitespace=True))
FRAME_MERGE_DATA = pandas.merge(FRAME_TR_TS, FRAME_IDENTITY, on='pic_name')
FRAME_MERGE_DATA = pandas.merge(FRAME_MERGE_DATA, FRAME_LANDMARK, on='pic_name')

'''
    DATA_STRUCTURE
        index   pic_name tag identity lefteye_x lefteye_y righteye_x righteye_y\
        nose_x  nose_y  leftmouth_x  leftmouth_y  rightmouth_x rightmouth_y
'''

FRAME_TRAIN1_INFO = FRAME_MERGE_DATA[FRAME_MERGE_DATA.tag == 0]
FRAME_TRAIN2_INFO = FRAME_MERGE_DATA[FRAME_MERGE_DATA.tag == 1]
FRAME_TEST_INFO = FRAME_MERGE_DATA[FRAME_MERGE_DATA.tag == 2]

LIST_TRAIN1_PEOPLE = list(FRAME_TRAIN1_INFO.groupby('identity').count().index)
# PIC_TRAIN1 = TRAIN1_INFO.pic_name.values
# PIC_TRAIN2 = TRAIN2_INFO.pic_name.values
# PIC_TEST = TEST_INFO.pic_name.values
ROOT_LOG.info(
    "TR and Ts was separated TR1:{} TR2:{} TS:{}".format(len(FRAME_TRAIN1_INFO), len(FRAME_TRAIN2_INFO),
                                                         len(FRAME_TEST_INFO)))

DATA_PARA_PEOPLE_LIMIT_TRAIN1 = min(len(LIST_TRAIN1_PEOPLE), 1000)
LIST_PEOPLE_LAYER1 = []
for index, people_name in enumerate(LIST_TRAIN1_PEOPLE):
    if index == DATA_PARA_PEOPLE_LIMIT_TRAIN1:
        break
    LIST_PEOPLE_LAYER1.append(people_name)
