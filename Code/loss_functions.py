import tensorflow as tf
import numpy as np

from tfutils import log10
import constants as c


def combined_loss(gen_frames, gt_frames, d_preds):
    """
    Calculates the sum of the combined adversarial, lp and GDL losses in the given proportion. Used
    for training the generative model.

    @param gen_frames: A list of tensors of the generated frames at each scale.
    @param gt_frames: A list of tensors of the ground truth frames at each scale.
    @param d_preds: A list of tensors of the classifications made by the discriminator model at each
                    scale.
    @param lam_adv: The percentage of the adversarial loss to use in the combined loss.
    @param lam_lp: The percentage of the lp loss to use in the combined loss.
    @param lam_gdl: The percentage of the GDL loss to use in the combined loss.
    @param l_num: 1 or 2 for l1 and l2 loss, respectively).
    @param alpha: The power to which each gradient term is raised in GDL loss.

    @return: The combined adversarial, lp and GDL losses.
    """
    batch_size = tf.shape(gen_frames[0])[0]  # variable batch size as a tensor
    loss_lp = lp_loss(gen_frames, gt_frames, c.L_NUM)
    loss_gdl = gdl_loss(gen_frames, gt_frames, c.ALPHA_NUM)
    loss_adv, _, _ = adv_loss(d_preds, tf.ones([batch_size, 1]))
    loss_global = c.LAM_LP * loss_lp
    loss_global += c.LAM_GDL * loss_gdl
    if c.ADVERSARIAL: loss_global += c.LAM_ADV * loss_adv

    return loss_global, loss_lp, loss_gdl, loss_adv


def bce_loss(preds, targets):
    """
    Calculates the sum of binary cross-entropy losses between predictions and ground truths.

    @param preds: A 1xN tensor. The predicted classifications of each frame.
    @param targets: A 1xN tensor The target labels for each frame. (Either 1 or -1). Not "truths"
                    because the generator passes in lies to determine how well it confuses the
                    discriminator.

    @return: The sum of binary cross-entropy losses.
    """
    batch_size = tf.shape(preds)[0]
    return tf.squeeze(-1 * (tf.matmul(targets, log10(preds), transpose_a=True) +
                     tf.matmul(1 - targets, log10(1 - preds), transpose_a=True))) / tf.cast(batch_size, tf.float32)


def lp_loss(gen_frames, gt_frames, l_num):
    """
    Calculates the sum of lp losses between the predicted and ground truth frames.

    @param gen_frames: The predicted frames at each scale.
    @param gt_frames: The ground truth frames at each scale
    @param l_num: 1 or 2 for l1 and l2 loss, respectively).

    @return: The lp loss.
    """
    # calculate the loss for each scale
    batch_size = tf.shape(gen_frames[0])[0]
    scale_losses = []
    for i in xrange(len(gen_frames)):
        scale_losses.append(tf.reduce_mean(tf.abs(gen_frames[i] - gt_frames[i])**l_num))

    # condense into one tensor and avg
    return tf.reduce_mean(tf.stack(scale_losses)) / tf.cast(batch_size, tf.float32)


def gdl_loss(gen_frames, gt_frames, alpha):
    """
    Calculates the sum of GDL losses between the predicted and ground truth frames.

    @param gen_frames: The predicted frames at each scale.
    @param gt_frames: The ground truth frames at each scale
    @param alpha: The power to which each gradient term is raised.

    @return: The GDL loss.
    """
    # calculate the loss for each scale
    batch_size = tf.shape(gen_frames[0])[0]
    scale_losses = []
    for i in xrange(len(gen_frames)):
        # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
        pos = tf.constant(np.identity(c.CHANNELS), dtype=tf.float32)
        neg = -1 * pos
        filter_x = tf.expand_dims(tf.stack([neg, pos]), 0)  # [-1, 1]
        filter_y = tf.stack([tf.expand_dims(pos, 0), tf.expand_dims(neg, 0)])  # [[1],[-1]]
        strides = [1, 1, 1, 1]  # stride of (1, 1)
        padding = 'SAME'

        gen_dx = tf.abs(tf.nn.conv2d(gen_frames[i], filter_x, strides, padding=padding))
        gen_dy = tf.abs(tf.nn.conv2d(gen_frames[i], filter_y, strides, padding=padding))
        gt_dx = tf.abs(tf.nn.conv2d(gt_frames[i], filter_x, strides, padding=padding))
        gt_dy = tf.abs(tf.nn.conv2d(gt_frames[i], filter_y, strides, padding=padding))

        grad_diff_x = tf.abs(gt_dx - gen_dx)
        grad_diff_y = tf.abs(gt_dy - gen_dy)

        scale_losses.append(tf.reduce_mean((grad_diff_x ** alpha + grad_diff_y ** alpha)))

    # condense into one tensor and avg
    return tf.reduce_mean(tf.stack(scale_losses)) / tf.cast(batch_size, tf.float32)


def adv_loss(preds, labels):
    """
    Calculates the sum of BCE losses between the predicted classifications and true labels.

    @param preds: The predicted classifications at each scale.
    @param labels: The true labels. (Same for every scale).

    @return: The adversarial loss.
    """
    # calculate the loss for each scale
    batch_size = tf.shape(labels[0])[0]
    scale_losses = []
    real_losses = []
    fake_losses = []
    for i in xrange(len(preds)):
        loss = bce_loss(preds[i], labels)
        scale_losses.append(loss)
        fake_loss = bce_loss(preds[i][:batch_size/2], labels[:batch_size/2])
        fake_losses.append(fake_loss)
        real_loss = bce_loss(preds[i][batch_size/2:], labels[batch_size/2:])
        real_losses.append(real_loss)
        scale_losses.append(loss)

    # condense into one tensor and avg
    av_loss = tf.reduce_mean(tf.stack(scale_losses))
    av_real_loss = tf.reduce_mean(tf.stack(real_losses))
    av_fake_loss = tf.reduce_mean(tf.stack(fake_losses))
    return av_loss, av_real_loss, av_fake_loss
