import numpy as np
import matplotlib.pyplot as plt

import constants as c
from utils import denormalize_frames


def display_batch(batch, n):

    channels = c.CHANNELS

    clip = np.squeeze(denormalize_frames(batch[n]))

    for fr in range(0, clip.shape[-1], channels):
        plt.figure(fr)
        cmap = 'gray' if channels == 1 else None
        plt.imshow(clip[:, :, fr:fr+channels], cmap=cmap)

    plt.show()
