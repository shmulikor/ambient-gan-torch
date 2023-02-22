# pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914, E1101

from __future__ import division

import numpy as np
import torch
import torch.nn.functional as F
from scipy import mgrid


#from commons import im_rotate


def get_gaussian_filter(radius, size):
    x, y = mgrid[-(size-1)/2:size/2, -(size-1)/2:size/2]
    g = np.exp(-(x**2/float(2*radius**2) + y**2/float(2*radius**2)))
    g = g / g.sum()
    return g


def blur(hparams, x):
    size = hparams['blur_filter_size']  # set size=1 for no blurring
    gaussian_filter = get_gaussian_filter(hparams['blur_radius'], size)
    gaussian_filter = torch.Tensor(np.reshape(gaussian_filter, [1, 1, size, size])).cuda()
    x_blurred_list = []
    for i in range(hparams['image_dims'][0]):
        x_blurred = F.conv2d(x[:, i:i + 1, :, :], gaussian_filter, stride=1, padding='same')
        x_blurred_list.append(x_blurred)
    x_blurred = torch.cat(x_blurred_list, dim=1)
    return x_blurred


def get_padding_ep(hparams):
    """Get padding for extract_patch measurements"""
    k = hparams['patch_size']
    if hparams['dataset'] == 'mnist':
        size = 32
    elif hparams['dataset'] == 'celebA':
        size = 64
    else:
        raise NotImplementedError
    pad_size = size - k
    paddings = (0, pad_size, pad_size, 0)
    return paddings


