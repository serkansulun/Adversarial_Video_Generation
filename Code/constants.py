import os
from scipy.ndimage import imread
import torch

EXPERIMENT_NAME = 'beyondmse'

EPOCHS = 100

LOAD_MODEL = None   # None to train from scratch

NUM_SCALE_NETS = 4  # number of scale nets

LRATE_D = 1e-6  # learning rate for the discriminator model
REG_D = 0   # regularization constant for the discriminator model
LRATE_G = 1e-4  # learning rate for the generator model

ADVERSARIAL = True  # whether to use adversarial training vs. basic training of the generator

LEAK = 0.2  # for leaky relu

BATCH_NORM = True   # only for the generator

BATCH_SIZE = 4  # the training minibatch size

TEST_ONLY = False
SAVE_MODEL = True

DATASET = 'KTH_frames'
CATEGORIES = ['handwaving']

CODE_DIR = os.getcwd()
MAIN_DIR = os.path.abspath(os.path.join(CODE_DIR, os.pardir))

# root directory for all data.
DATA_DIR = os.path.join(MAIN_DIR, 'Data/KTH_frames')

TRAIN_DIR = os.path.join(DATA_DIR, 'Train/')
TEST_DIR = os.path.join(DATA_DIR, 'Test/')


def list_paths(folder, path=True):
    if path:
        return sorted([os.path.join(folder, file_) for file_ in os.listdir(folder)])
    else:
        return sorted(os.listdir(folder))


def get_dimensions():
    video_path = list_paths(TEST_DIR)[0]
    frame_path = list_paths(video_path)[0]
    frame = imread(frame_path)
    return frame.shape


HEIGHT, WIDTH, CHANNELS = get_dimensions()
CHANNELS = 1    # override for KTH

##
# Data
##

CUDA = torch.cuda.is_available()
if CUDA:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
TORCH3 = torch.__version__[:3] != '0.4'

HISTORY_FRAMES = [0, 1, 2, 3]     # frames to use as history
PRED_FRAMES = [4]       # frames to use as prediction (ground-truth)

# the number of history frames to give as input to the network
HIST_LEN = len(HISTORY_FRAMES)
# the number of predicted frames as output of the network
PRED_LEN = len(PRED_FRAMES)

# sequence length in minibatch
SEQ_LEN = HIST_LEN + PRED_LEN
# number of channels in minibatch
CHANNEL_LEN = SEQ_LEN * CHANNELS

def makedir(directory):
    """
    Creates the given directory if it does not exist.

    @param directory: The path to the directory.
    @return: The path to the directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

# directory for saved models
MODEL_DIR = makedir(os.path.join(MAIN_DIR, 'Models/'))

# directory for saved images
IMG_SAVE_DIR = makedir(os.path.join(MAIN_DIR, 'Images/'))

##
# General training
##

##
# Loss parameters
##

L_NUM = 2   # for lp loss. e.g, 1 or 2 for l1 and l2 loss, respectively)
ALPHA_NUM = 1   # the power to which each gradient term is raised in GDL loss
LAMBDAS = [1, 1, 0.05]     # [LP, GD, ADV] loss coefficients

##
# Generator model
##

# feature maps for each convolution of each scale network in the generator model
# e.g SCALE_FMS_G[1][2] is the input of the 3rd convolution in the 2nd scale network.

SCALE_FMS_G = [[CHANNELS * HIST_LEN, 128, 256, 128, CHANNELS * PRED_LEN],
               [CHANNELS * (HIST_LEN + PRED_LEN), 128, 256, 128, CHANNELS * PRED_LEN],
               [CHANNELS * (HIST_LEN + PRED_LEN), 128, 256, 512, 256, 128, CHANNELS * PRED_LEN],
               [CHANNELS * (HIST_LEN + PRED_LEN), 128, 256, 512, 256, 128, CHANNELS * PRED_LEN]]
# kernel sizes for each convolution of each scale network in the generator model
SCALE_KERNEL_SIZES_G = [[3, 3, 3, 3],
                        [5, 3, 3, 5],
                        [5, 3, 3, 3, 3, 5],
                        [7, 5, 5, 5, 5, 7]]

##
# Discriminator model
##

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

if NUM_SCALE_NETS < 4:
    # if user wants to use less scale nets, take nets from the highest scales
    SCALE_FMS_G = SCALE_FMS_G[:NUM_SCALE_NETS]
    SCALE_FMS_G[0][0] = CHANNELS * HIST_LEN
    SCALE_KERNEL_SIZES_G = SCALE_KERNEL_SIZES_G[:NUM_SCALE_NETS]
    SCALE_CONV_FMS_D = SCALE_CONV_FMS_D[:NUM_SCALE_NETS]
    SCALE_KERNEL_SIZES_D = SCALE_KERNEL_SIZES_D[:NUM_SCALE_NETS]
    SCALE_FC_LAYER_SIZES_D = SCALE_FC_LAYER_SIZES_D[:NUM_SCALE_NETS]

# Padding to achieve an output size same as input

SCALE_PADDING_SIZES_D = [[(kernel-1)/2 for kernel in scale] for scale in SCALE_KERNEL_SIZES_D]
SCALE_PADDING_SIZES_G = [[(kernel-1)/2 for kernel in scale] for scale in SCALE_KERNEL_SIZES_G]

#################### delete lines below ############################

##
# Generator model
##

# feature maps for each convolution of each scale network in the generator model
# e.g SCALE_FMS_G[1][2] is the input of the 3rd convolution in the 2nd scale network.

SCALE_FMS_G = [[CHANNELS * HIST_LEN, 1, CHANNELS * PRED_LEN],
               [CHANNELS * (HIST_LEN + PRED_LEN), 1, CHANNELS * PRED_LEN],
               [CHANNELS * (HIST_LEN + PRED_LEN), 1, CHANNELS * PRED_LEN],
               [CHANNELS * (HIST_LEN + PRED_LEN), 1, CHANNELS * PRED_LEN]]
# kernel sizes for each convolution of each scale network in the generator model
SCALE_KERNEL_SIZES_G = [[3,3],
                        [5,3],
                        [5,3],
                        [7,3]]

##
# Discriminator model
##

# feature maps for each convolution of each scale network in the discriminator model
SCALE_CONV_FMS_D = [[CHANNELS * (HIST_LEN + PRED_LEN), 1],
                    [CHANNELS * (HIST_LEN + PRED_LEN), 1],
                    [CHANNELS * (HIST_LEN + PRED_LEN), 1],
                    [CHANNELS * (HIST_LEN + PRED_LEN), 1]]

# kernel sizes for each convolution of each scale network in the discriminator model
SCALE_KERNEL_SIZES_D = [[3],
                        [3],
                        [5],
                        [7]]

# layer sizes for each fully-connected layer of each scale network in the discriminator model
# layer connecting conv to fully-connected is dynamically generated when creating the model
SCALE_FC_LAYER_SIZES_D = [[1],
                          [1],
                          [1],
                          [1]]

if NUM_SCALE_NETS < 4:
    # if user wants to use less scale nets, take nets from the highest scales
    SCALE_FMS_G = SCALE_FMS_G[:NUM_SCALE_NETS]
    SCALE_FMS_G[0][0] = CHANNELS * HIST_LEN
    SCALE_KERNEL_SIZES_G = SCALE_KERNEL_SIZES_G[:NUM_SCALE_NETS]
    SCALE_CONV_FMS_D = SCALE_CONV_FMS_D[:NUM_SCALE_NETS]
    SCALE_KERNEL_SIZES_D = SCALE_KERNEL_SIZES_D[:NUM_SCALE_NETS]
    SCALE_FC_LAYER_SIZES_D = SCALE_FC_LAYER_SIZES_D[:NUM_SCALE_NETS]

# Padding to achieve an output size same as input

SCALE_PADDING_SIZES_D = [[(kernel-1)/2 for kernel in scale] for scale in SCALE_KERNEL_SIZES_D]
SCALE_PADDING_SIZES_G = [[(kernel-1)/2 for kernel in scale] for scale in SCALE_KERNEL_SIZES_G]
