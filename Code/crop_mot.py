import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import configparser
from myutils import makedir, list_paths, num2str

config = configparser.ConfigParser()

height_min = 128
width_min = int(round(height_min / 3))
visibility_min = 1

dataset_path = '../Data/MOT17'

videos_path = os.path.join(dataset_path, 'train')

videos_list = list_paths(videos_path)

crops_folder = os.path.join(dataset_path, 'crops', 'train')

if not os.path.exists(crops_folder):
    makedir(crops_folder)

for vid_ind, video_path in enumerate(videos_list):
    print 'Remaining: ', len(videos_list)-vid_ind

    config.read(os.path.join(video_path, 'seqinfo.ini'))
    fps = config.get('Sequence', 'frameRate')
    #period = 1000.0 / fps    # in miliseconds

    frames_folder = os.path.join(video_path, 'img1')
    frames_list = list_paths(frames_folder)
    gt = np.genfromtxt(os.path.join(video_path, 'gt/gt.txt'), delimiter=',')

    for fr_ind, frame_path in enumerate(frames_list):

        gt_fr = gt[gt[:, 0] == fr_ind+1, :]
        frame = Image.open(frame_path)
        for gt_obj in gt_fr:

            [obj_ind, x1, y1, width, height, care, class_ind], visibility = gt_obj[1:-1].astype('int'), gt_obj[-1]

            if care == 1 and visibility >= visibility_min and class_ind == 1 and width >= width_min and height >= height_min:

                tile = (x1, y1, x1 + width, y1 + height)
                cropped = frame.crop(tile)

                vid_number = video_path[-8:-6]
                obj_number = num2str(obj_ind, 3)
                frame_number = int(frame_path[-10:-4])
                #time = int(round((frame_number - 1) * period))

                crop_obj_folder = os.path.join(crops_folder, 'vid' + vid_number + '_obj' + obj_number + '_' + fps + 'fps')
                makedir(crop_obj_folder)

                crop_path = os.path.join(crop_obj_folder, num2str(frame_number, 4) + '.png')
                cropped.save(crop_path)


