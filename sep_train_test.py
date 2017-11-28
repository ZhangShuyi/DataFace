import os
import random
import pandas as pd

DATA_PATH = 'F:\\CAS-PEAL-R1-64_64_feature'
SAVE_PATH = 'F:\\CAS-PEAL-R1-64_64_feature.csv'
SAVE_PATH_TEST = 'F:\\CAS-PEAL-R1-64_64_feature_ts.csv'
SEP_NUM = 5

MODEL_LIST = [ \
    '0076', '0028', '0070', \
    '0024', '0005', '0050', \
    '0051', '0008', \
    '0033', '0031', '0060', \
    '0047', '0057', \
    '0058', '0068', '0054']


def sepTrainData(data_path, sep_num):
    csv_data = []
    csv_data_test = []
    people_list = os.listdir(data_path)
    for people_name in people_list:
        test_list = []
        for i, feature_num in enumerate(os.listdir(os.path.join(data_path, people_name))):
            read_file_path = os.path.join(data_path, people_name, feature_num)
            if i == 0:
                test_list = random.sample(range(len(os.listdir(read_file_path))), sep_num)
            for j, picture in enumerate(os.listdir(read_file_path)):
                picture_num = int(picture[7:11])
                if j in test_list:
                    csv_data_test.append(
                        [people_name, int(feature_num), picture_num, os.path.join(people_name, feature_num, picture)])
                else:
                    csv_data.append(
                        [people_name, int(feature_num), picture_num, os.path.join(people_name, feature_num, picture)])
        print("Info: {} was finished".format(people_name))
    df = pd.DataFrame(data=csv_data, columns=["people", "feature", "picture_num", "path"])
    df_test = pd.DataFrame(data=csv_data_test, columns=["people", "feature", "picture_num", "path"])
    return df, df_test


if __name__ == "__main__":
    data, data_test = sepTrainData(data_path=DATA_PATH, sep_num=SEP_NUM)
    data.to_csv(SAVE_PATH)
    data_test.to_csv(SAVE_PATH_TEST)
