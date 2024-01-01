# pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914, E1101

"""Implementations of measurement"""

from __future__ import division

import copy
import torch.nn.functional as F
import cv2
import numpy as np
import torch
from scipy import signal

from commons import measure_utils


def get_mdevice(hparams):
    if hparams['measurement_type'] == 'drop_independent':
        mdevice = DropIndependent(hparams)
    elif hparams['measurement_type'] == 'drop_row':
        mdevice = DropRow(hparams)
    elif hparams['measurement_type'] == 'drop_col':
        mdevice = DropCol(hparams)
    elif hparams['measurement_type'] == 'drop_rowcol':
        mdevice = DropRowCol(hparams)
    elif hparams['measurement_type'] == 'drop_patch':
        mdevice = DropPatch(hparams)
    elif hparams['measurement_type'] == 'keep_patch':
        mdevice = KeepPatch(hparams)
    elif hparams['measurement_type'] == 'extract_patch':
        mdevice = ExtractPatch(hparams)
    elif hparams['measurement_type'] == 'blur_addnoise':
        mdevice = BlurAddNoise(hparams)
    else:
        raise NotImplementedError
    return mdevice


class MeasurementDevice(object):
    """Base class for measurement devices"""

    def __init__(self, hparams):
        # self.measurement_type = hparams.measurement_type
        self.batch_dims = [hparams['batch_size']] + hparams['image_dims']
        self.output_type = None  # indicate whether image or vector

    def sample_theta(self, hparams, labels=None):
        """Abstract Method"""
        # Should return theta_val
        raise NotImplementedError

    def measure(self, hparams, x, theta):
        """Abstract Method"""
        # Tensorflow implementation of measurement. Must be differentiable wrt x.
        # Should return x_measured
        raise NotImplementedError


class DropDevice(MeasurementDevice):

    def __init__(self, hparams):
        MeasurementDevice.__init__(self, hparams)
        self.output_type = 'image'

    def sample_theta(self, hparams, labels=None):
        """Abstract Method"""
        # Should return theta_val
        raise NotImplementedError

    def measure(self, hparams, x, theta):
        if hparams['dataset'] == 'mnist' or hparams['dataset'] == 'celebA':
            x_measured = torch.mul(theta, x.mul(0.5).add(0.5)).mul(2).add(-1) # only for mnist and celebA
        else:
            x_measured = torch.mul(theta, x) # for the 3D case
        return x_measured


class DropMaskType1(DropDevice):

    def get_noise_shape(self):
        """Abstract Method"""
        # Should return noise_shape
        raise NotImplementedError

    def sample_theta(self, hparams, labels=None):
        noise_shape = self.get_noise_shape()
        repeat_axis = np.ones(len(noise_shape) - 1).astype(int)
        repeat_axis[0] = noise_shape[1]
        mask = torch.Tensor(np.stack([torch.rand(noise_shape[2:]).repeat(tuple(repeat_axis))
                                      for _ in range(noise_shape[0])])) # sample the same mask for all channels
        p = hparams['drop_prob']
        mask = (mask >= p).to(torch.float32) #/ (1 - p)
        return mask


class DropIndependent(DropMaskType1):

    def get_noise_shape(self):
        noise_shape = copy.deepcopy(self.batch_dims)
        return noise_shape


class DropRow(DropMaskType1):

    def get_noise_shape(self):
        noise_shape = copy.deepcopy(self.batch_dims)
        noise_shape[2] = 1
        return noise_shape


class DropCol(DropMaskType1):

    def get_noise_shape(self):
        noise_shape = copy.deepcopy(self.batch_dims)
        noise_shape[1] = 1
        return noise_shape


class DropRowCol(DropDevice):

    def sample_theta(self, hparams, labels=None):
        drop_row = DropRow(hparams)
        mask1 = drop_row.sample_theta(hparams)
        drop_col = DropCol(hparams)
        mask2 = drop_col.sample_theta(hparams)
        theta_val = mask1 * mask2
        return theta_val


class DropMaskType2(DropDevice):

    def sample_theta(self, hparams, labels=None):
        raise NotImplementedError

    def patch_mask(self, hparams):
        k = hparams['patch_size']
        h, w = hparams['image_dims'][1:3]
        patch_mask = np.ones(self.batch_dims)
        for i in range(hparams['batch_size']):
            x, y = np.random.choice(h-k), np.random.choice(w-k)
            patch_mask[i, :, x:x+k, y:y+k] = 0
        return torch.Tensor(patch_mask)


class DropPatch(DropMaskType2):

    def sample_theta(self, hparams, labels=None):
        return self.patch_mask(hparams)


class KeepPatch(DropMaskType2):

    def sample_theta(self, hparams, labels=None):
        return 1 - self.patch_mask(hparams)


class ExtractPatch(MeasurementDevice):

    def __init__(self, hparams):
        MeasurementDevice.__init__(self, hparams)
        self.output_type = 'image'

    def sample_theta(self, hparams, labels=None):
        k = hparams['patch_size']
        h, w = hparams['image_dims'][1:3]
        theta = np.zeros([hparams['batch_size'], 2])
        for i in range(hparams['batch_size']):
            x, y = np.random.choice(h-k), np.random.choice(w-k)
            theta[i, :] = [x, y]
        return torch.IntTensor(theta)

    def measure(self, hparams, x, theta):
        k = hparams['patch_size']
        patch_list = []
        for t in range(hparams['batch_size']):
            i, j = theta[t, 0], theta[t, 1]
            patch = x[t, :, i:i+k, j:j+k]
            patch = torch.reshape(patch, [1, hparams['image_dims'][0], k, k])
            patch_list.append(patch)
        patches = torch.cat(patch_list, dim=0)
        paddings = measure_utils.get_padding_ep(hparams)
        x_measured = F.pad(patches, paddings, mode='constant', value=-1)
        return x_measured

class BlurAddNoise(MeasurementDevice):

    def __init__(self, hparams):
        MeasurementDevice.__init__(self, hparams)
        self.output_type = 'image'

    def sample_theta(self, hparams, labels=None):
        theta_val = hparams['additive_noise_std'] * torch.randn(*(self.batch_dims))
        return theta_val

    def measure(self, hparams, x, theta):
        x_blurred = measure_utils.blur(hparams, x)
        x_measured = torch.add(x_blurred, theta)
        return x_measured
