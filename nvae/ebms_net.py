import torch.nn as nn
from collections import OrderedDict

class netEzi(nn.Module):

    def __init__(self, z_shape, args):
        super().__init__()
        self.z_shape = z_shape
        self.nef = args.nef
        self.ndf = args.ndf
        layers = {8: 1, 16: 2, 32: 3, 64: 4, 128: 5}

        b, c, h, w = z_shape[0], z_shape[1], z_shape[2], z_shape[3]
        num_layers = layers[h]

        convlayers = OrderedDict()
        current_dims = c
        for i in range(num_layers):
            convlayers['conv{}'.format(i+1)] = nn.Conv2d(current_dims, self.nef, 4, 2, 1)
            convlayers['lrelu{}'.format(i+1)] = nn.LeakyReLU(0.2)
            current_dims = self.nef
        self.conv2d = nn.Sequential(convlayers)

        linearlayers = OrderedDict()
        current_dims = self.nef*4*4
        num_layers = num_layers
        for i in range(num_layers): # just try num layers, could be other.
            linearlayers['fc{}'.format(i+1)] = nn.Linear(current_dims, self.ndf)
            linearlayers['lrelu{}'.format(i+1)] = nn.LeakyReLU(0.2)
            current_dims = self.ndf

        linearlayers['out'] = nn.Linear(current_dims, 1)
        self.linear = nn.Sequential(linearlayers)

    def forward(self, z):
        assert z.shape[1:] == self.z_shape[1:]
        f = self.conv2d(z)
        f = f.view(z.shape[0], self.nef*4*4)
        en = self.linear(f)
        return en.squeeze(1)
