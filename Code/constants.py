import numpy as np
import os
from glob import glob
import shutil
from datetime import datetime
from scipy.ndimage import imread

##
# Data
##

def get_latest_model():
    """

    :return: Path of the latest model
    """
    global LOAD_MODEL
    folder = os.path.join(TF_DIR, 'Models/', LOAD_MODEL)
    files = sorted(os.listdir(folder))
    file_name = files[-1][:-5]

    return os.path.join(folder, file_name)

def get_date_str():
    """
    @return: A string representing the current date/time that can be used as a directory name.
    """
    return str(datetime.now()).replace(' ', '_').replace(':', '.')[:-10]

def get_dir(directory):
    """
    Creates the given directory if it does not exist.

    @param directory: The path to the directory.
    @return: The path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def clear_dir(directory):
    """
    Removes all files in the given directory.

    @param directory: The path to the directory.
    """
    for f in os.listdir(directory):
        path = os.path.join(directory, f)
        try:
            if os.path.isfile(path):
                os.unlink(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        except Exception as e:
            print(e)

def get_test_frame_dims():
    img_path = glob(os.path.join(TEST_DIR, '*/*'))[0]
    if CHANNELS == 1:
        img = imread(img_path, mode='L')
        img = np.expand_dims(img, axis=-1)
    else:
        img = imread(img_path, mode='RGB')
    shape = np.shape(img)

    return shape[0], shape[1]

def get_train_frame_dims():
    img_path = glob(os.path.join(TRAIN_DIR, '*/*'))[0]
    if CHANNELS == 1:
        img = imread(img_path, mode='L')
        img = np.expand_dims(img, axis=-1)
    else:
        img = imread(img_path, mode='RGB')
    shape = np.shape(img)

    return shape[0], shape[1]

def set_test_dir(directory):
    """
    Edits all constants dependent on TEST_DIR.

    @param directory: The new test directory.
    """
    global TEST_DIR, FULL_HEIGHT, FULL_WIDTH

    TEST_DIR = directory
    FULL_HEIGHT, FULL_WIDTH = get_test_frame_dims()


EXPERIMENT = 'MOT_6_7'   # inner directory to differentiate between runs
INFO = 'MOT gen 1e-6 disc 1e-7'
LOAD_MODEL = EXPERIMENT   # None to train from scratch
# root directory for all data.
DATA_DIR = get_dir('../Data/MOT17/crops')
CHANNELS = 3


# directory of unprocessed training frames
TRAIN_DIR = os.path.join(DATA_DIR, 'train/')
# directory of unprocessed test frames

# Directory of processed training clips.
# hidden so finder doesn't freeze w/ so many files. DON'T USE `ls` COMMAND ON THIS DIR!
PROCESSED_DIR = get_dir(os.path.join(DATA_DIR, '.sequences/'))
PROCESSED_TRAIN_DIR = get_dir(os.path.join(PROCESSED_DIR, 'train/'))
PROCESSED_TEST_DIR = get_dir(os.path.join(PROCESSED_DIR, 'test/'))
TEST_DIR = PROCESSED_TEST_DIR

# For processing clips. l2 diff between frames must be greater than this
MOVEMENT_THRESHOLD = 400
# total number of processed clips in TRAIN_DIR_CLIPS
NUM_CLIPS = len(glob(TRAIN_DIR + '*'))

# the height and width of the full frames to test on. Set in avg_runner.py or process_data.py main.
FULL_HEIGHT = 120
FULL_WIDTH = int(round(FULL_HEIGHT / 3))
# the height and width of the patches to train on
TRAIN_HEIGHT = FULL_HEIGHT
TRAIN_WIDTH = FULL_WIDTH

HISTORY_TIME = [0, 150]     # approximate miliseconds of frames to use as history
PRED_TIME = [300]       # approximate time of predicted frame

# the number of history frames to give as input to the network
HIST_LEN = len(HISTORY_TIME)    # 8
# the number of predicted frames as output of the network
PRED_LEN = len(PRED_TIME)
# number of test recursions
NUM_TEST_REC = 1

# sequence length in minibatch
SEQ_LEN = HIST_LEN + PRED_LEN
# number of channels in minibatch
CHANNEL_LEN = SEQ_LEN * CHANNELS


##
# Output
##

def set_save_name():
    """
    Edits all constants dependent on SAVE_NAME.

    @param name: The new save name.
    """
    global EXPERIMENT, MODEL_SAVE_DIR, SUMMARY_SAVE_DIR, IMG_SAVE_DIR

    MODEL_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'Models/', EXPERIMENT))
    SUMMARY_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'Summaries/', EXPERIMENT))
    IMG_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'Images/', EXPERIMENT))

def clear_save_name():
    """
    Clears all saved content for SAVE_NAME.
    """
    clear_dir(MODEL_SAVE_DIR)
    clear_dir(SUMMARY_SAVE_DIR)
    clear_dir(IMG_SAVE_DIR)


SAVE_DIR = '../Save/'
TF_DIR = '../tf/'   # TF related
# directory for saved models
MODEL_SAVE_DIR = get_dir(os.path.join(TF_DIR, 'Models/', EXPERIMENT))
if LOAD_MODEL is not None:
    # path for model to load
    MODEL_LOAD_PATH = get_latest_model()
# directory for saved TensorBoard summaries
SUMMARY_SAVE_DIR = get_dir(os.path.join(TF_DIR, 'Summaries/', EXPERIMENT))
# directory for saved images
IMG_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'Images/', EXPERIMENT))
PERFORMANCE_SAVE_DIR = get_dir(os.path.join(SAVE_DIR, 'Performance/'))

EPOCHS = 1000

SUMMARY_FREQ    = 100  # how often to save the summaries, in # steps
IMG_SAVE_FREQ   = 1000  # how often to save generated images, in # steps
TEST_FREQ       = float('inf')  # 1000   # how often to test the model on test data, in # steps
STATS_FREQ      = 100     # how often to print loss/train error stats, in # steps
MODEL_SAVE_FREQ = 5000  # how often to save the model, in # steps
MODEL_KEEP_FREQ = 4 * MODEL_SAVE_FREQ  # 25000  # how often to keep the model, in # steps

##
# General training
##

# whether to use adversarial training vs. basic training of the generator
ADVERSARIAL = True
# the training minibatch size
BATCH_SIZE = 4

##
# Loss parameters
##

# for lp loss. e.g, 1 or 2 for l1 and l2 loss, respectively)
L_NUM = 2
# the power to which each gradient term is raised in GDL loss
ALPHA_NUM = 1
# the percentage of the adversarial loss to use in the combined loss
LAM_ADV = 1
# the percentage of the lp loss to use in the combined loss
LAM_LP = 15
# the percentage of the GDL loss to use in the combined loss
LAM_GDL = 10

##
# Generator model
##

# learning rate for the generator model
LRATE_G = 1e-6  # Value in paper is 0.04
# padding for convolutions in the generator model
PADDING_G = 'SAME'
# feature maps for each convolution of each scale network in the generator model
# e.g SCALE_FMS_G[1][2] is the input of the 3rd convolution in the 2nd scale network.
SCALE_FMS_G = [[CHANNELS * HIST_LEN, 128, 256, 128, CHANNELS],
               [CHANNELS * (HIST_LEN + 1), 128, 256, 128, CHANNELS],
               [CHANNELS * (HIST_LEN + 1), 128, 256, 512, 256, 128, CHANNELS],
               [CHANNELS * (HIST_LEN + 1), 128, 256, 512, 256, 128, CHANNELS]]
# kernel sizes for each convolution of each scale network in the generator model
SCALE_KERNEL_SIZES_G = [[3, 3, 3, 3],
                        [5, 3, 3, 5],
                        [5, 3, 3, 3, 3, 5],
                        [7, 5, 5, 5, 5, 7]]


##
# Discriminator model
##

# learning rate for the discriminator model
LRATE_D = 1e-7    # 0.02
# padding for convolutions in the discriminator model
PADDING_D = 'VALID'
# feature maps for each convolution of each scale network in the discriminator model
SCALE_CONV_FMS_D = [[CHANNELS * (HIST_LEN + PRED_LEN), 64],
                    [CHANNELS * (HIST_LEN + PRED_LEN), 64, 128, 128],
                    [CHANNELS * (HIST_LEN + PRED_LEN), 128, 256, 256],
                    [CHANNELS * (HIST_LEN + PRED_LEN), 128, 256, 512, 128]]
# kernel sizes for each convolution of each scale network in the discriminator model
SCALE_KERNEL_SIZES_D = [[3],
                        [3, 3, 3],
                        [5, 5, 5],
                        [7, 7, 5, 5]]
# layer sizes for each fully-connected layer of each scale network in the discriminator model
# layer connecting conv to fully-connected is dynamically generated when creating the model
SCALE_FC_LAYER_SIZES_D = [[512, 256, 1],
                          [1024, 512, 1],
                          [1024, 512, 1],
                          [1024, 512, 1]]
