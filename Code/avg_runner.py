import tensorflow as tf
import getopt
import sys
import os
import cPickle as pickle
import numpy as np
import time
import datetime

from utils import get_train_batch, testBatcher
import constants as c
from g_model import GeneratorModel
from d_model import DiscriminatorModel
from performance import Performance
from batch import Batcher
from debug import display_batch
from myutils import list_paths

class AVGRunner:
    def __init__(self, num_steps):

        """
        Initializes the Adversarial Video Generation Runner.

        @param num_steps: The number of training steps to run.
        @param model_load_path: The path from which to load a previously-saved model.
                                Default = None.
        @param num_test_rec: The number of recursive generations to produce when testing. Recursive
                             generations use previous generations as input to predict further into
                             the future.
        """

        self.global_step = 0
        self.num_steps = num_steps
        self.num_test_rec = c.NUM_TEST_REC

        self.sess = tf.Session()
        self.summary_writer = tf.summary.FileWriter(c.SUMMARY_SAVE_DIR, graph=self.sess.graph)

        if c.ADVERSARIAL:
            print 'Init discriminator...'
            self.d_model = DiscriminatorModel(self.sess,
                                              self.summary_writer,
                                              c.TRAIN_HEIGHT,
                                              c.TRAIN_WIDTH,
                                              c.SCALE_CONV_FMS_D,
                                              c.SCALE_KERNEL_SIZES_D,
                                              c.SCALE_FC_LAYER_SIZES_D)

        print 'Init generator...'
        self.g_model = GeneratorModel(self.sess,
                                      self.summary_writer,
                                      c.TRAIN_HEIGHT,
                                      c.TRAIN_WIDTH,
                                      c.FULL_HEIGHT,
                                      c.FULL_WIDTH,
                                      c.SCALE_FMS_G,
                                      c.SCALE_KERNEL_SIZES_G)

        self.summary_writer = tf.summary.FileWriter(c.SUMMARY_SAVE_DIR, graph=self.sess.graph)

        print 'Init variables...'
        self.saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)
        self.sess.run(tf.global_variables_initializer())

        # if load path specified, load a saved model
        if c.LOAD_MODEL is not None:
            self.saver.restore(self.sess, c.MODEL_LOAD_PATH)
            print 'Model restored from ' + c.MODEL_LOAD_PATH

    def train(self):
        """
        Runs a training loop on the model networks.
        """
        performance_saver = Performance()


        remove_model = None

        once = True

        for epoch in xrange(c.EPOCHS):
            epoch_done = False
            step = 0
            t1 = time.time()
            f = open('remaining.txt', 'w')
            f.write('EPOCH: ' + str(epoch) + '\n')
            f.close()
            batcher_disc = Batcher('train')
            dataset_size = batcher_disc.dataset_size()
            batcher_gen = Batcher('train')
            save_img = True
            while not epoch_done:

                if c.ADVERSARIAL:
                    # update discriminator
                    try:
                        batch, disc_done = batcher_disc.get()
                    except Exception as e:
                        print(e)
                        print 'Read error: ', batcher_disc.path
                        continue
                    # print 'Training discriminator...'
                    _, disc_loss_batch, disc_real_loss_batch, disc_fake_loss_batch = self.d_model.train_step(batch, self.g_model)
                # update generator
                try:
                    batch, gen_done = batcher_gen.get()
                except Exception as e:
                    print(e)
                    print 'Read error: ', batcher_disc.path
                    continue
                # print 'Training generator...'
                self.global_step, gen_performance = self.g_model.train_step(
                    batch, save_img=save_img, discriminator=(self.d_model if c.ADVERSARIAL else None))
                disc_performance = {'disc_global_loss': disc_loss_batch, 'disc_real_loss': disc_real_loss_batch,
                                    'disc_fake_loss': disc_fake_loss_batch}
                performance_all = dict(disc_performance.items() + gen_performance.items())
                performance_saver.update(performance_all)

                epoch_done = disc_done or gen_done

                step += 1
                if step % 100 == 0:
                    t2 = time.time()
                    f = open('remaining.txt', 'a')
                    f.write(str((t2 - t1)/60) + '\t' + str(batcher_disc.remaining()) + '\n')
                    print(str(t2 - t1) + '\t' + str(batcher_disc.remaining()) + '\n')
                    f.close()

                if once:
                    np.save('done', [])
                    print 'First batch done'
                    np.save('done', [])
                    once = False



            # test generator model
            #if self.global_step % c.TEST_FREQ == 0:
            tst_psnr_av, tst_sharpdiff_av = self.test()
            # else:
            #     tst_psnr_av, tst_sharpdiff_av = None, None

            #if self.global_step % c.STATS_FREQ == 0:
            performance_saver.save(dataset_size, tst_psnr_av=tst_psnr_av, tst_sharpdiff_av=tst_sharpdiff_av)
                #performance_saver.save()

            # save the models
            #if self.global_step % c.MODEL_SAVE_FREQ == 0:
                # print '-' * 30
                # print 'Saving models...'

            self.saver.save(self.sess,
                            c.MODEL_SAVE_DIR + '/model.ckpt',
                            global_step=epoch)

                # if remove_model:
                #     os.remove(c.MODEL_SAVE_DIR + '/model.ckpt-' + str(remove_model) + '.data-00000-of-00001')
                #     os.remove(c.MODEL_SAVE_DIR + '/model.ckpt-' + str(remove_model) + '.index')
                #     os.remove(c.MODEL_SAVE_DIR + '/model.ckpt-' + str(remove_model) + '.meta')
                #
                # if self.global_step % c.MODEL_KEEP_FREQ == 0:
                #     remove_model = None
                # else:
                #     remove_model = self.global_step


    def test(self):
        """
        Runs one test step on the generator network.
        """
        psnr = np.zeros(c.NUM_TEST_REC)
        sharpdiff = np.zeros(c.NUM_TEST_REC)
        batch_count = 0
        batcher_test = Batcher('test')
        save_imgs = True
        test_done = False
        while not test_done:
            batch, test_done = batcher_test.get()
            batch_psnr, batch_sharpdiff = self.g_model.test_batch(batch, self.global_step, save_imgs=save_imgs)
            psnr += batch_psnr
            sharpdiff += batch_sharpdiff
            batch_count += 1
            save_imgs = False

        psnr /= batch_count
        sharpdiff /= batch_count

        return psnr, sharpdiff


def usage():
    print 'Options:'
    print '-l/--load_path=    <Relative/path/to/saved/model>'
    print '-t/--test_dir=     <Directory of test images>'
    print '-r/--recursions=   <# recursive predictions to make on test>'
    print '-a/--adversarial=  <{t/f}> (Whether to use adversarial training. Default=True)'
    print '-n/--name=         <Subdirectory of ../Data/Save/*/ in which to save output of this run>'
    print '-s/--steps=        <Number of training steps to run> (Default=1000001)'
    print '-O/--overwrite     (Overwrites all previous data for the model with this save name)'
    print '-T/--test_only     (Only runs a test step -- no training)'
    print '-H/--help          (Prints usage)'
    print '--stats_freq=      <How often to print loss/train error stats, in # steps>'
    print '--summary_freq=    <How often to save loss/error summaries, in # steps>'
    print '--img_save_freq=   <How often to save generated images, in # steps>'
    print '--test_freq=       <How often to test the model on test data, in # steps>'
    print '--model_save_freq= <How often to save the model, in # steps>'


def main():
    ##
    # Handle command line input.
    ##

    test_only = False
    num_test_rec = c.NUM_TEST_REC  # number of recursive predictions to make on test
    num_steps = 100000
    try:
        opts, _ = getopt.getopt(sys.argv[1:], 'l:t:r:a:n:s:OTH',
                                ['load_path=', 'test_dir=', 'recursions=', 'adversarial=', 'name=',
                                 'steps=', 'overwrite', 'test_only', 'help', 'stats_freq=',
                                 'summary_freq=', 'img_save_freq=', 'test_freq=',
                                 'model_save_freq='])
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-l', '--load_path'):
            c.LOAD_MODEL = arg
        if opt in ('-t', '--test_dir'):
            c.set_test_dir(arg)
        if opt in ('-r', '--recursions'):
            num_test_rec = int(arg)
        if opt in ('-a', '--adversarial'):
            c.ADVERSARIAL = (arg.lower() == 'true' or arg.lower() == 't')
        # if opt in ('-n', '--name'):
        #     c.set_save_name(arg)
        if opt in ('-s', '--steps'):
            num_steps = int(arg)
        if opt in ('-O', '--overwrite'):
            c.clear_save_name()
        if opt in ('-H', '--help'):
            usage()
            sys.exit(2)
        if opt in ('-T', '--test_only'):
            test_only = True
        if opt == '--stats_freq':
            c.STATS_FREQ = int(arg)
        if opt == '--summary_freq':
            c.SUMMARY_FREQ = int(arg)
        if opt == '--img_save_freq':
            c.IMG_SAVE_FREQ = int(arg)
        if opt == '--test_freq':
            c.TEST_FREQ = int(arg)
        if opt == '--model_save_freq':
            c.MODEL_SAVE_FREQ = int(arg)

    # set test frame dimensions
    assert os.path.exists(c.TEST_DIR)
    #.FULL_HEIGHT, c.FULL_WIDTH = c.get_test_frame_dims()

    ##
    # Init and run the predictor
    ##

    runner = AVGRunner(num_steps)
    if test_only:
        runner.test()
    else:
        runner.train()


if __name__ == '__main__':
    main()
