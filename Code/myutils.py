import os
import numpy as np
from skimage.transform import resize
from skimage.util import pad
from math import floor, ceil

import constants as c


def pad_to_ratio(img, ratio=c.FULL_HEIGHT / c.FULL_WIDTH):
    cur_ratio = 1.0 * img.shape[0] / img.shape[1]
    if cur_ratio > ratio:
        new_width = int(round(1.0 * img.shape[0] / ratio))
        add_to_width = new_width - img.shape[1]
        left = int(floor(add_to_width / 2.0))
        right = int(ceil(add_to_width / 2.0))
        up = 0
        down = 0
    else:
        new_height = int(round(1.0 * img.shape[1] * ratio))
        add_to_height = new_height - img.shape[0]
        up = int(floor(add_to_height / 2.0))
        down = int(ceil(add_to_height / 2.0))
        left = 0
        right = 0
    img_padded = np.pad(img, [[up, down], [left, right], [0, 0]], mode='edge')
    return img_padded



def batch_resize(batch, height, width):
    batch_out = np.zeros((batch.shape[0], height, width, batch.shape[-1]))
    if -1 <= np.min(batch) <= 0:
        img_min = -1
    else:
        img_min = 0
    if np.max(batch) <= 1:
        img_max = 1
    else:
        img_max = 255
    img_range = img_max - img_min

    for i, sequence in enumerate(batch):
        for j in range(0, batch.shape[-1], c.CHANNELS):
            frame = sequence[:, :, j:j + c.CHANNELS]
            frame = (frame - img_min) / img_range
            frame = resize(frame, [height, width, frame.shape[-1]])
            batch_out[i, :, :, j:j + c.CHANNELS] = frame * img_range + img_min

    return batch_out

def num2str(num, n):
    # Pads zeros and converts to string
    string = str(num)
    while len(string) < n:
        string = '0' + string
    return string


def makedir(directory):
    """
    Creates the given directory if it does not exist.

    @param directory: The path to the directory.
    @return: The path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


def list_paths(folder):
    return sorted([os.path.join(folder, file_) for file_ in os.listdir(folder)])