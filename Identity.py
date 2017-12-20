from InternalModule.Envs import *



class Identity:
    def __init__(self, identity):
        self.identity = identity

    def load_picture(self):
        self.data_frame = FRAME_MERGE_DATA[FRAME_MERGE_DATA.identity == self.identity]
        self.pic_name_list = list(self.data_frame['pic_name'])
        print(self.pic_name_list)

    def get_batch(self):
        batchoperation = SepBatch()