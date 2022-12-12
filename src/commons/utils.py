from __future__ import division

import os

import torch
import torch.utils.data
from commons import basic_utils
from commons import dir_def


def setup_vals(hparams):
    """Setup some values in hparams"""
    if hparams['dataset'] == 'mnist':
        hparams['c_dim'] = 1
        hparams['image_dims'] = [1, 32, 32]
        hparams['train_size'] = 60000
        hparams['y_dim'] = 10  # [Optional] Number of labels
        hparams['x_min'] = 0
        hparams['x_max'] = 1
        hparams['signal_power'] = 0.11204461  # Assuming each pixel in [0, 1]
    elif hparams['dataset'] == 'celebA':
        hparams['c_dim'] = 3
        hparams['image_dims'] = [3, 64, 64]
        hparams['train_size'] = 180000
        hparams['x_min'] = -1
        hparams['x_max'] = 1
        hparams['signal_power'] = 0.2885201  # Assuming each pixel in [-1, 1]
    # elif hparams['dataset'] == 'cifar10':
    #     hparams['c_dim'] = 3
    #     hparams['image_dims'] = [3, 32, 32]
    #     hparams['train_size'] = 50000
    #     hparams['y_dim'] = 10  # [Optional] Number of labels
    elif hparams['dataset'] == 'QSM':
        hparams['c_dim'] = 64
        hparams['image_dims'] = [1, 64, 64, 64]
    else:
        raise NotImplementedError

    expt_dir = dir_def.get_expt_dir(hparams)
    hparams['hparams_dir'] = hparams['results_dir'] + 'hparams/' + expt_dir
    hparams['summary_dir'] = hparams['results_dir'] + 'summ/' + expt_dir
    hparams['sample_dir'] = hparams['results_dir'] + 'samples/' + expt_dir
    hparams['ckpt_dir'] = hparams['results_dir'] + 'ckpt/' + expt_dir
    hparams['incpt_dir'] = hparams['results_dir'] + 'incpt/' + expt_dir
    hparams['log_dir'] = hparams['results_dir'] + 'logs/' + expt_dir
    hparams['incpt_pkl'] = hparams['incpt_dir'] + 'score.pkl'


def setup_dirs(hparams):
    """Setup the dirs"""
    basic_utils.set_up_dir(hparams['hparams_dir'])
    basic_utils.set_up_dir(hparams['ckpt_dir'])
    basic_utils.set_up_dir(hparams['summary_dir'])
    basic_utils.set_up_dir(hparams['sample_dir'])
    basic_utils.set_up_dir(hparams['incpt_dir'])


def sample_z_val(hparams, device):
    if hparams['z_dist'] == 'uniform':
        z = -2 * torch.rand(hparams['batch_size'], hparams['c_dim'], 1, 1, device=device) + 1  # uniform over [-1,1]
    elif hparams['z_dist'] == 'gaussian':
        z = torch.randn(hparams['batch_size'], hparams['c_dim'], 1, 1, device=device)
    else:
        raise NotImplementedError
    return z



