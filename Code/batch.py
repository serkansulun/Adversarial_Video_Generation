from random import shuffle
import numpy as np
import torch
from scipy.ndimage import imread
from utils import scale_batch, num2str, normalize_frames
import constants as c
from constants import list_paths


def load_sequence(first_file):
    sequence = np.zeros((c.HEIGHT, c.WIDTH, c.CHANNELS * (c.HIST_LEN + c.PRED_LEN)))
    frame_number = int(first_file[-8:-4])
    for ind, fr in enumerate(c.HISTORY_FRAMES + c.PRED_FRAMES):
        next_frame_number = frame_number + ind
        next_file = first_file[:-8] + num2str(next_frame_number, 4) + first_file[-4:]
        frame = imread(next_file, mode='L')
        frame = np.expand_dims(frame, axis=-1)
        sequence[:, :, fr * c.CHANNELS: (fr + 1) * c.CHANNELS] = frame

    sequence = normalize_frames(sequence)
    return sequence

def prepare(batch):
    batch_scaled_np = scale_batch(batch)

    batch_scaled, history, gt = [], [], []
    for i in range(c.NUM_SCALE_NETS):
        batch_single_scale = torch.from_numpy(batch_scaled_np[i])
        batch_single_scale.requires_grad = False
        if c.CUDA:
            batch_single_scale = batch_single_scale.cuda()
        batch_scaled.append(batch_single_scale)
        history.append(batch_scaled[i][:, :c.CHANNELS * c.HIST_LEN, :, :])
        gt.append(batch_scaled[i][:, c.CHANNELS * c.HIST_LEN:, :, :])

    return history, gt

class Batcher:

    def __init__(self, mode):

        if mode == 'train':
            dataset_path = c.TRAIN_DIR
        elif mode == 'test':
            dataset_path = c.TEST_DIR

        self.mode = mode

        self.frames_list = []  # list of first frames
        all_videos = list_paths(dataset_path)

        self.videos = all_videos

        for video in self.videos:
            for category in c.CATEGORIES:
                if category in video:
                    frames = list_paths(video)
                    self.frames_list += frames[:1 - c.HIST_LEN - c.PRED_LEN]

        self.new()

    def new(self, shuffle_samples=True):
        if shuffle_samples:
            shuffle(self.frames_list)
        self.ind = 0

    def get(self):

        if self.ind + c.BATCH_SIZE >= len(self.frames_list):
            batch_size = len(self.frames_list) - self.ind
            done = True
        else:
            batch_size = c.BATCH_SIZE
            done = False

        batch = np.zeros((batch_size, c.HEIGHT, c.WIDTH, c.CHANNEL_LEN))
        for ind_sequence in range(batch_size):
            path = self.frames_list[self.ind]

            batch[ind_sequence] = load_sequence(path)
            self.ind += 1

        batch = np.transpose(batch, axes=(0, 3, 1, 2))
        batch = batch.astype(np.float32)

        history, gt = prepare(batch)

        return history, gt, done
