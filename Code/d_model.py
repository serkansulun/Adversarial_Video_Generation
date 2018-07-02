import torch.nn as nn
import torch

from utils import  conv_out_size, Flatten
import constants as c


class DiscriminatorModel(nn.Module):
    def __init__(self):
        super(DiscriminatorModel, self).__init__()

        self.d_model = []
        for scale_num in xrange(c.NUM_SCALE_NETS):

            scale_factor = 1. / 2 ** ((c.NUM_SCALE_NETS - 1) - scale_num)
            scale_model = self.d_single_scale_model(scale_num, int(c.HEIGHT * scale_factor), int(c.WIDTH * scale_factor))

            self.d_model.append(scale_model)

        self.d_model = nn.Sequential(*self.d_model)

    def d_single_scale_model(self, scale_index, height, width):
        """
        Sets up the model graph in TensorFlow.
        """

        ##
        # Layer setup
        ##

        # convolution
        ws = []
        fc_layer_sizes = c.SCALE_FC_LAYER_SIZES_D[scale_index]

        last_out_height = height
        last_out_width = width
        for i in xrange(len(c.SCALE_KERNEL_SIZES_D[scale_index])):
            ws.append(nn.Conv2d(c.SCALE_CONV_FMS_D[scale_index][i], c.SCALE_CONV_FMS_D[scale_index][i + 1],
                                kernel_size=c.SCALE_KERNEL_SIZES_D[scale_index][i],
                                padding=c.SCALE_PADDING_SIZES_D[scale_index][i]))
            ws.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=1))
            ws.append(nn.LeakyReLU(c.LEAK))

            last_out_height = conv_out_size(input=last_out_height, padding=c.SCALE_PADDING_SIZES_D[scale_index][i],
                                            kernel=c.SCALE_KERNEL_SIZES_D[scale_index][i], stride=1)

            last_out_width = conv_out_size(
                last_out_width, c.SCALE_PADDING_SIZES_D[scale_index][i], c.SCALE_KERNEL_SIZES_D[scale_index][i], 1)

            last_out_height = conv_out_size(input=last_out_height, kernel=2, padding=1, stride=2)
            last_out_width = conv_out_size(input=last_out_width, kernel=2, padding=1, stride=2)

        ws.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=1))

        last_out_height = conv_out_size(input=last_out_height, kernel=2, padding=1, stride=2)
        last_out_width = conv_out_size(input=last_out_width, kernel=2, padding=1, stride=2)

        # fully-connected

        # Add in an initial layer to go from the last conv to the first fully-connected.
        # Use /2 for the height and width because there is a 2x2 pooling layer
        fc_layer_sizes.insert(0, last_out_height * last_out_width * c.SCALE_CONV_FMS_D[scale_index][-1])

        ws.append(Flatten())

        for i in xrange(len(fc_layer_sizes) - 1):
            ws.append(nn.Linear(fc_layer_sizes[i], fc_layer_sizes[i + 1]))
            if i == len(fc_layer_sizes) - 2:
                ws.append(nn.Sigmoid())
            else:
                ws.append(nn.LeakyReLU(c.LEAK, inplace=True))

        d_single_scale = nn.Sequential(*ws)

        return d_single_scale

    def forward(self, input_):

        scale_preds = []
        for scale_num in xrange(c.NUM_SCALE_NETS):
            # get predictions from the scale network
            single_scale_pred = self.d_model[scale_num](input_[scale_num])
            single_scale_pred = torch.clamp(single_scale_pred, 0.001, 0.999)    # for stability
            scale_preds.append(single_scale_pred)

        return scale_preds

