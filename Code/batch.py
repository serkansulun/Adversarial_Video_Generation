from random import shuffle
import numpy as np

import constants as c
from myutils import list_paths


class Batcher:

    def __init__(self, mode):

        path = c.PROCESSED_TRAIN_DIR if mode == 'train' else c.PROCESSED_TEST_DIR
        self.files_list = []
        for folder in list_paths(path):
            self.files_list += list_paths(folder)

        shuffle(self.files_list)
        self.ind = 0

    def get(self):

        if self.ind + c.BATCH_SIZE >= len(self.files_list):
            batch_size = len(self.files_list) - self.ind
            done = True
        else:
            batch_size = c.BATCH_SIZE
            done = False

        batch = np.zeros((batch_size, c.FULL_HEIGHT, c.FULL_WIDTH, c.CHANNEL_LEN))
        for ind_sequence in range(batch_size):
            path = self.files_list[self.ind]
            batch[ind_sequence] = np.load(path)['arr_0']
            self.ind += 1

        return batch, done


    def remaining(self):
        return len(self.files_list) - self.ind

    def dataset_size(self):
        return len(self.files_list)




