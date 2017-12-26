import pickle
from InternalModule.Envs import *
from  SepBatch import *

BATCH_NUM = 5


class OnePic:
    def __init__(self, pic_name):
        self.pic_name = pic_name
        landmark_data = FRAME_MERGE_DATA[FRAME_MERGE_DATA.pic_name == self.pic_name]
        assert len(landmark_data) == 1
        self.identity = int(landmark_data.identity)
        self.img = ReadPicture(os.path.join(PATH_DATA_ALIGNED, self.pic_name), color=PARA.COLOR, to_array=False)
        self.left_eye = [int(landmark_data.lefteye_x), int(landmark_data.lefteye_y)]
        self.right_eye = [int(landmark_data.righteye_x), int(landmark_data.righteye_y)]
        self.left_mouth = [int(landmark_data.leftmouth_x), int(landmark_data.leftmouth_y)]
        self.right_mouth = [int(landmark_data.rightmouth_x), int(landmark_data.rightmouth_y)]
        self.nose = [int(landmark_data.nose_x), int(landmark_data.nose_y)]
        self.size = self.img.shape
        self.max_index = 3

    def getIndexBatch(self, index):
        if index == 0:
            return self.shearPictureV1(self.left_eye[0], self.left_eye[1], self.right_eye[0], self.right_mouth[1], 20,
                                       20)
        if index == 1:
            return self.shearPictureV1(self.left_eye[0], self.left_eye[1], self.right_eye[0], self.nose[1], 10, 10)

        if index == 2:
            return self.shearPictureV1(self.left_mouth[0], self.nose[1], self.right_mouth[0], self.right_mouth[1], 10,
                                       10)

    def getAllBatch(self):
        self.batches = []
        for i in range(self.max_index):
            self.batches.append(self.getIndexBatch(i))
        return self.batches

    def shearPictureV1(self, x1, y1, x2, y2, x_add, y_add):
        x1 = max(0, x1 - x_add)
        y1 = max(0, y1 - y_add)
        x2 = min(self.size[0], x2 + x_add)
        y2 = min(self.size[1], y2 + x_add)
        return cv2.resize(self.img[y1:y2, x1:x2], dsize=(DATA_PARA_D_SIZE, DATA_PARA_D_SIZE),
                          interpolation=cv2.INTER_CUBIC)

    def showBatch(self):
        print(self.batches)
        fig = plt.figure()
        total = len(self.batches)
        width = 2
        height = total / width + 1
        for index, img in enumerate(self.batches):
            print(img.shape)
            img_r = img[:, :, 0]
            img_g = img[:, :, 1]
            img_b = img[:, :, 2]
            img = cv2.merge([img_b, img_g, img_r])
            ax = fig.add_subplot(width, height, index + 1)
            ax.imshow(img)
        fig.show()
        fig.waitforbuttonpress()


class Identity:
    def __init__(self, identity):
        self.identity = identity

    def loadInfo(self):
        self.data_frame = FRAME_MERGE_DATA[FRAME_MERGE_DATA.identity == self.identity]
        self.pic_name_list = list(self.data_frame['pic_name'])

    def saveIndexBatch(self, index, save_path):
        # self.current_batch_dict = {}
        # for pic_name in self.pic_name_list:
        #     self.current_batch_dict[pic_name] = OnePic(pic_name).getIndexBatch(index)
        # with open(save_path, 'wb') as file:
        #     pickle.dump(self.current_batch_dict, file)
        b_dict = {}
        r_dict = {}
        g_dict = {}
        for pic_name in self.pic_name_list:
            b, r, g = cv2.split(OnePic(pic_name).getIndexBatch(index))
            b_dict[pic_name] = b
            r_dict[pic_name] = r
            g_dict[pic_name] = g

        if not os.path.exists(save_path + '_b'):
            os.mkdir(save_path + '_b')
        if not os.path.exists(save_path + '_r'):
            os.mkdir(save_path + '_r')
        if not os.path.exists(save_path + '_g'):
            os.mkdir(save_path + '_g')
        with open(os.path.join(save_path + '_b', str(self.identity)), 'wb') as file:
            pickle.dump(b_dict, file)
        with open(os.path.join(save_path + '_r', str(self.identity)), 'wb') as file:
            pickle.dump(r_dict, file)
        with open(os.path.join(save_path + '_g', str(self.identity)), 'wb') as file:
            pickle.dump(g_dict, file)

    def readIndexBatch(self, save_path):
        with open(save_path, 'rb') as file:
            self.current_batch_dict = pickle.load(file)
        return self.current_batch_dict

    def showCurrentBatch(self):
        fig = plt.figure()
        for index, pic_name in enumerate(self.current_batch_dict):
            img = self.current_batch_dict[pic_name]
            ax = fig.add_subplot(5, 10, index + 1)
            ax.imshow(img)
        fig.show()
        fig.waitforbuttonpress()


class OneBatch:
    def __init__(self, name, identity_list, batch_index):
        self.name = name
        self.batch_index = batch_index
        self.identity_list = identity_list
        self.path = os.path.join(PATH_DATA_BATCHES, self.name)
        if not os.path.exists(self.path):
            os.mkdir(self.path)

    def saveData(self):
        for people_name in self.identity_list:
            I = Identity(people_name)
            I.loadInfo()
            I.saveIndexBatch(self.batch_index, self.path)
            PRINT_LOG.info("save batch {} Identity {} data finished".format(self.name, people_name))

    def loadOneIdentity(self, people_name):
        I = Identity(people_name)
        dict = I.readIndexBatch(os.path.join(self.path, str(people_name)))
        PRINT_LOG.info(
            "Batch index {} people: {} data was loaded total num {}".format(self.batch_index, people_name, len(dict)))
        return dict


if __name__ == "__main__":
    OneBatch("batch0_64_64_3", LIST_PEOPLE_LAYER1, 0).saveData()
    OneBatch('batch1_top_64_64_3', LIST_PEOPLE_LAYER1, 1).saveData()
    OneBatch('batch2_bottom_64_64_3', LIST_PEOPLE_LAYER1, 2).saveData()

    OneBatch("batch0_64_64_3", LIST_TRAIN2_PEOPLE, 0).saveData()
    OneBatch('batch1_top_64_64_3', LIST_TRAIN2_PEOPLE, 1).saveData()
    OneBatch('batch2_bottom_64_64_3', LIST_TRAIN2_PEOPLE, 2).saveData()

    OneBatch("batch0_64_64_3", LIST_TEST_PEOPLE, 0).saveData()
    OneBatch('batch1_top_64_64_3', LIST_TEST_PEOPLE, 1).saveData()
    OneBatch('batch2_bottom_64_64_3', LIST_TEST_PEOPLE, 2).saveData()
