from random import shuffle
import numpy as np
from scipy.ndimage import imread
from skimage.transform import resize
from os.path import join
from copy import deepcopy

import constants as c
from myutils import list_paths, pad_to_ratio, makedir, num2str
from utils import normalize_frames

from bisect import bisect_left


def take_closest_ind(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return 0 # myList[0]
    if pos == len(myList):
        return len(myList) # myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
       return pos #after
    else:
       return pos-1 #before


class Sequencer:

    def __init__(self, mode='train'):

        path = c.TRAIN_DIR if mode == 'train' else c.TEST_DIR
        self.savedir = makedir(c.PROCESSED_TRAIN_DIR)

        self.vid_folders_list = list_paths(path)

        self.ind_tuples = []
        for vid_ind, vid_folder in enumerate(self.vid_folders_list):
            frames_list = list_paths(vid_folder)
            for fr_ind in range(len(frames_list)):
                self.ind_tuples.append((vid_ind, fr_ind))

        self.delays = c.HISTORY_TIME + c.PRED_TIME
        self.sequence_length = len(c.HISTORY_TIME) + len(c.PRED_TIME)

    def run(self):

        for i, ind_tuple in enumerate(self.ind_tuples):
            next_files = self.get_next_sequence_files(ind_tuple)
            if next_files:
                self.save(next_files)

            if i % 100 == 0:
                print 'Remaining: ', len(self.ind_tuples) - i

    def get_next_sequence_files(self, ind_tuple):

        (vid_ind, fr_ind) = ind_tuple
        vid_folder = self.vid_folders_list[vid_ind]
        fps = int(vid_folder[-5:-3])

        frame_delays = self.get_frame_delays(fps)

        episode_frame_names = list_paths(vid_folder)
        episode_frame_numbers = [int(file_name[-8:-4]) for file_name in episode_frame_names]

        first_frame_name = episode_frame_names[fr_ind]
        first_frame_number = episode_frame_numbers[fr_ind]   # TODO

        sequence_frame_numbers = [frame_delay + first_frame_number for frame_delay in frame_delays]

        contains = set(sequence_frame_numbers).issubset(episode_frame_numbers)
        sequence_files = []
        if contains:
            for frame_number in sequence_frame_numbers:
                file_name = join(vid_folder, num2str(frame_number, 4)) + '.png'
                sequence_files.append(file_name)

        return sequence_files

    def get_frame_delays(self, fps):
        period = 1000.00 / fps
        timestamps = [period * i for i in range(0, 20)]
        frame_delays = []
        for delay in self.delays:
            frame_delays.append(take_closest_ind(timestamps, delay))
        return frame_delays

    def save(self, file_names):

        vid_obj_name = file_names[0][-27:-15]
        folder = makedir(join(self.savedir, vid_obj_name))

        sequence = np.zeros((c.FULL_HEIGHT, c.FULL_WIDTH, c.CHANNEL_LEN))

        for ind_frame, frame_name in enumerate(file_names):
            if c.CHANNELS == 1:
                frame = imread(frame_name, mode='L')
                frame = np.expand_dims(frame, axis=-1)
            else:
                frame = imread(frame_name, mode='RGB')
            norm_frame = normalize_frames(frame)
            pad_frame = pad_to_ratio(norm_frame)
            resized_frame = resize(pad_frame, (c.FULL_HEIGHT, c.FULL_WIDTH), order=3)
            # clips[clip_num, :, :, frame_num * c.CHANNELS:(frame_num + 1) * c.CHANNELS] = norm_frame
            sequence[:, :, ind_frame * c.CHANNELS: (ind_frame + 1) * c.CHANNELS] = resized_frame

        npy_file_name = join(folder, vid_obj_name + '_' + file_names[0][-8:-4])    # TODO
        np.savez_compressed(npy_file_name, sequence)

    def remaining(self):
        return len(self.ind_tuples) - self.ind


sequencer = Sequencer()
sequencer.run()

