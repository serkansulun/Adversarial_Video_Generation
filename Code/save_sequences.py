import numpy as np
from os.path import join


from sequence import Sequencer
from myutils import makedir
import constants as c

savedir = c.PROCESSED_TRAIN_DIR
makedir(savedir)
i = 0
done = False
sequencer = Sequencer('train')
while not done:
    batch, name, done = sequencer.get()
    np.savez_compressed(join(savedir, name + '_' + str(i)), batch)
    i += 1
    if i % 100 == 0:
        print sequencer.remaining()