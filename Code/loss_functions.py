import numpy as np
import torch.nn.functional as F
import torch

from pytorch_msssim import ssim
import constants as c


def combined_loss(gen_frames, gt_frames, preds=None):
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

    batch_size = gen_frames[0].size()[0]  # variable batch size as a tensor
    loss_lp = lp_loss(gen_frames, gt_frames, c.L_NUM)
    loss_gdl = gdl_loss(gen_frames, gt_frames)

    loss_global = c.LAMBDAS[0] * loss_lp
    loss_global += c.LAMBDAS[1] * loss_gdl
    if c.ADVERSARIAL:
        loss_adv = adv_loss(preds, torch.ones([batch_size, 1]))
        loss_global += c.LAMBDAS[2] * loss_adv

    return loss_global


def bce_loss(preds, targets):
    """
    Calculates the sum of binary cross-entropy losses between predictions and ground truths.

    @param preds: A 1xN tensor. The predicted classifications of each frame.
    @param targets: A 1xN tensor The target labels for each frame. (Either 1 or -1). Not "truths"
                    because the generator passes in lies to determine how well it confuses the
                    discriminator.

    @return: The sum of binary cross-entropy losses.
    """
    return F.binary_cross_entropy(preds, targets)



def lp_loss(gen_frames, gt_frames, l_num):
    """
    Calculates the sum of lp losses between the predicted and ground truth frames.

    @param gen_frames: The predicted frames at each scale.
    @param gt_frames: The ground truth frames at each scale
    @param l_num: 1 or 2 for l1 and l2 loss, respectively).

    @return: The lp loss.
    """
    # calculate the loss for each scale
    # batch_size = torch.shape(gen_frames[0])[0]
    scale_losses = []
    for i in xrange(len(gen_frames)):
        scale_losses.append(torch.mean(torch.abs(gen_frames[i] - gt_frames[i])**l_num))

    # condense into one tensor and avg
    return torch.mean(torch.stack(scale_losses))


def gdl_loss(gen_frames, gt_frames):
    """
    Calculates the sum of GDL losses between the predicted and ground truth frames.

    @param gen_frames: The predicted frames at each scale.
    @param gt_frames: The ground truth frames at each scale
    @param alpha: The power to which each gradient term is raised.

    @return: The GDL loss.
    """
    # calculate the loss for each scale
    # batch_size = torch.shape(gen_frames[0])[0]
    scale_losses = []
    for i in xrange(len(gen_frames)):
        # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
        pos = torch.from_numpy(np.identity(c.CHANNELS, dtype=np.float32))
        if c.CUDA:
            pos = pos.cuda()
        neg = -1 * pos
        filter_x = torch.stack([neg, pos], dim=2).unsqueeze(2)  # [-1, 1]
        filter_y = torch.stack([neg, pos], dim=2).unsqueeze(-1)  # [[1],[-1]]

        gen_dx = torch.abs(F.conv2d(gen_frames[i], filter_x))
        gen_dy = torch.abs(F.conv2d(gen_frames[i], filter_y))
        gt_dx = torch.abs(F.conv2d(gt_frames[i], filter_x))
        gt_dy = torch.abs(F.conv2d(gt_frames[i], filter_y))

        grad_diff_x = torch.abs(gt_dx - gen_dx)
        grad_diff_y = torch.abs(gt_dy - gen_dy)

        scale_losses.append(torch.mean(grad_diff_x ** c.ALPHA_NUM) + torch.mean(grad_diff_y ** c.ALPHA_NUM))

    # condense into one tensor and avg
    return torch.mean(torch.stack(scale_losses))


def adv_loss(preds, labels):
    """
    Calculates the sum of BCE losses between the predicted classifications and true labels.

    @param preds: The predicted classifications at each scale.
    @param labels: The true labels. (Same for every scale).

    @return: The adversarial loss.
    """
    # calculate the loss for each scale
    batch_size = labels.size()[0]
    scale_losses = []

    for i in xrange(len(preds)):
        loss = bce_loss(preds[i], labels)
        scale_losses.append(loss)

    # condense into one tensor and avg
    av_loss = torch.mean(torch.stack(scale_losses))
    return av_loss
