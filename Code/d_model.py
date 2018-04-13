import tensorflow as tf
import numpy as np
from skimage.transform import resize

from d_scale_model import DScaleModel
from loss_functions import adv_loss
import constants as c
from myutils import batch_resize
from debug import display_batch

# noinspection PyShadowingNames
class DiscriminatorModel:
    def __init__(self, session, summary_writer, height, width, scale_conv_layer_fms,
                 scale_kernel_sizes, scale_fc_layer_sizes):
        """
        Initializes a DiscriminatorModel.

        @param session: The TensorFlow session.
        @param summary_writer: The writer object to record TensorBoard summaries
        @param height: The height of the input images.
        @param width: The width of the input images.
        @param scale_conv_layer_fms: The number of feature maps in each convolutional layer of each
                                     scale network.
        @param scale_kernel_sizes: The size of the kernel for each layer of each scale network.
        @param scale_fc_layer_sizes: The number of nodes in each fully-connected layer of each scale
                                     network.

        @type session: tf.Session
        @type summary_writer: tf.train.SummaryWriter
        @type height: int
        @type width: int
        @type scale_conv_layer_fms: list<list<int>>
        @type scale_kernel_sizes: list<list<int>>
        @type scale_fc_layer_sizes: list<list<int>>
        """
        self.sess = session
        self.summary_writer = summary_writer
        self.height = height
        self.width = width
        self.scale_conv_layer_fms = scale_conv_layer_fms
        self.scale_kernel_sizes = scale_kernel_sizes
        self.scale_fc_layer_sizes = scale_fc_layer_sizes
        self.num_scale_nets = len(scale_conv_layer_fms)

        self.define_graph()

    # noinspection PyAttributeOutsideInit
    def define_graph(self):
        """
        Sets up the model graph in TensorFlow.
        """
        with tf.name_scope('discriminator'):
            ##
            # Setup scale networks. Each will make the predictions for images at a given scale.
            ##

            self.scale_nets = []
            for scale_num in xrange(self.num_scale_nets):
                with tf.name_scope('scale_net_' + str(scale_num)):
                    scale_factor = 1. / 2 ** ((self.num_scale_nets - 1) - scale_num)
                    self.scale_nets.append(DScaleModel(scale_num,
                                                       int(self.height * scale_factor),
                                                       int(self.width * scale_factor),
                                                       self.scale_conv_layer_fms[scale_num],
                                                       self.scale_kernel_sizes[scale_num],
                                                       self.scale_fc_layer_sizes[scale_num]))

            # A list of the prediction tensors for each scale network
            self.scale_preds = []
            for scale_num in xrange(self.num_scale_nets):
                self.scale_preds.append(self.scale_nets[scale_num].preds)

            ##
            # Data
            ##

            self.labels = tf.placeholder(tf.float32, shape=[None, 1], name='labels')


            ##
            # Training
            ##

            with tf.name_scope('training'):
                # global loss is the combined loss from every scale network
                self.global_loss, self.real_loss, self.fake_loss = adv_loss(self.scale_preds, self.labels)
                self.global_step = tf.Variable(0, trainable=False, name='global_step')
                self.optimizer = tf.train.AdamOptimizer(c.LRATE_D, name='optimizer')
                self.train_op = self.optimizer.minimize(self.global_loss,
                                                        global_step=self.global_step,
                                                        name='train_op')

                # add summaries to visualize in TensorBoard
                loss_summary = tf.summary.scalar('loss_D', self.global_loss)
                self.summaries = tf.summary.merge([loss_summary])

    def build_feed_dict(self, batch, generator):
        """
        Builds a feed_dict with resized inputs and outputs for each scale network.

        @param input_frames: An array of shape
                             [batch_size x self.height x self.width x (c.CHANNELS * HIST_LEN)], The frames to
                             use for generation.
        @param gt_output_frames: An array of shape [batch_size x self.height x self.width x c.CHANNELS], The
                                 ground truth outputs for each sequence in input_frames.
        @param generator: The generator model.

        @return: The feed_dict needed to run this network, all scale_nets, and the generator
                 predictions.
        """
        feed_dict = {}
        real_feed_dict = {}
        fake_feed_dict = {}
        batch_size = np.shape(batch)[0]

        ##
        # Split into inputs and outputs
        ##

        input_frames = batch[:, :, :, :-c.CHANNELS]
        gt_output_frames = batch[:, :, :, -c.CHANNELS:]

        ##
        # Get generated frames from GeneratorModel
        ##

        g_feed_dict = {generator.input_frames_train: input_frames,
                       generator.gt_frames_train: gt_output_frames}
        g_scale_preds = self.sess.run(generator.scale_preds_train, feed_dict=g_feed_dict)

        ##
        # Create discriminator feed dict
        ##
        for scale_num in xrange(self.num_scale_nets):
            scale_net = self.scale_nets[scale_num]

            # resize sequence frames
            # scaled_sequence_frames = np.empty([batch_size, scale_net.height, scale_net.width, batch.shape[-1]])
            # for i, fr in enumerate(batch):
            #     for j in range(0, batch.shape[-1], c.CHANNELS):
            #         img = batch[i, :, :, j:j+c.CHANNELS]
            #         scaled_sequence_frames[i, :, :, j:j + c.CHANNELS] = imresize(img, scale_net.width, scale_net.height)
                    # for skimage.transform.resize, images need to be in range [0, 1], so normalize to
                    # [0, 1] before resize and back to [-1, 1] after
                    # sknorm_img = (img / 2) + 0.5
                    # resized_frame = resize(sknorm_img, [scale_net.height, scale_net.width, c.CHANNELS])
                    # scaled_sequence_frames[i, :, :, j:j+c.CHANNELS] = (resized_frame - 0.5) * 2
            scaled_sequence_frames = batch_resize(batch, scale_net.height, scale_net.width)
            scaled_hist_frames = scaled_sequence_frames[:, :, :, :c.HIST_LEN * c.CHANNELS]

            # concatenate gt history and predicted next frames to create fake sample for discriminator
            gen_sequence = np.concatenate((scaled_hist_frames, g_scale_preds[scale_num]), axis=-1)

            # combine with resized gt_output_frames to get inputs for prediction
            scaled_input_frames = np.concatenate([gen_sequence, scaled_sequence_frames])

            # convert to np array and add to feed_dict
            feed_dict[scale_net.input_frames] = scaled_input_frames
            real_feed_dict[scale_net.input_frames] = scaled_sequence_frames
            fake_feed_dict[scale_net.input_frames] = gen_sequence

        # add labels for each image to feed_dict
        batch_size = np.shape(input_frames)[0]
        feed_dict[self.labels] = np.concatenate([np.zeros([batch_size, 1]),
                                                 np.ones([batch_size, 1])])
        real_feed_dict[self.labels] = np.ones([batch_size, 1])
        fake_feed_dict[self.labels] = np.zeros([batch_size, 1])


        return feed_dict, real_feed_dict, fake_feed_dict

    def train_step(self, batch, generator):
        """
        Runs a training step using the global loss on each of the scale networks.

        @param batch: An array of shape
                      [BATCH_SIZE x self.height x self.width x (c.CHANNELS * (HIST_LEN + 1))]. The input
                      and output frames, concatenated along the channel axis (index 3).
        @param generator: The generator model.

        @return: The global step.
        """

        ##
        # Train
        ##

        feed_dict, real_feed_dict, fake_feed_dict = self.build_feed_dict(batch, generator)

        _, global_loss, real_loss, fake_loss, global_step, summaries, scale_preds, labels = self.sess.run(
            [self.train_op, self.global_loss, self.real_loss, self.fake_loss, self.global_step, self.summaries, self.scale_preds, self.labels],
            feed_dict=feed_dict)



        ##
        # User output
        ##

        #print np.matrix(np.asarray(scale_preds))
        if global_step % c.STATS_FREQ == 0:
            print 'DiscriminatorModel: step %d | global loss: %f' % (global_step, global_loss)

        if global_step % c.SUMMARY_FREQ == 0:
            # print 'DiscriminatorModel: saved summaries'
            self.summary_writer.add_summary(summaries, global_step)

        return global_step, global_loss, real_loss, fake_loss
