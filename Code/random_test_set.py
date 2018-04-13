from os.path import join
import constants as c
from myutils import makedir, list_paths
from random import sample
from shutil import move

source_dir = c.PROCESSED_TRAIN_DIR
dest_dir = makedir(c.PROCESSED_TEST_DIR)

sequences = list_paths(source_dir)
total_size = len(sequences)

ratio = 0.1

move_size = int(round(ratio * total_size))

folders_to_move = sample(sequences, move_size)

for folder in folders_to_move:
    move(folder, dest_dir)
