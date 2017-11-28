from VerNet import VERNET
'''
SAMPLE_PATH = "/media/dcs/Elements/FeatureID_t"
MODEL_PATH = "/media/dcs/Elements/Final_Model"
PIC_NAME_PATH = "/home/dcs/Desktop/Test_data"
'''

SAMPLE_PATH = "//home/dcs/Desktop/FeatureID"
MODEL_PATH = "/media/dcs/Elements/Final_Model"
PIC_NAME_PATH = "/home/dcs/Desktop/CAS-PEAL-R1-64_64"

MODEL_LIST = [ \
    '0000', '0028', '0070', \
    '0024', '0005', '0050', \
    '0051', '0008', \
    '0033', '0031', '0060', \
    '0047', '0057', \
    '0058', '0068', '0054' \
    ]

V = VERNET(model_list=MODEL_LIST, model_feature_size=100, name="Verification")
V.build_model()
V.restore_para(MODEL_PATH)
V.load_csv(SAMPLE_PATH)
V.LoadTestAndTrainDict(PIC_NAME_PATH, train_ratio=0, test_ratio=1)
V.load_sample_TS(10000, pic_name_path=PIC_NAME_PATH)

print(V.get_accuracy(10000))
