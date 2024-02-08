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
    elif hparams['dataset'] == 'celebA':
        hparams['c_dim'] = 3
        hparams['image_dims'] = [3, 64, 64]
    elif hparams['dataset'] == 'QSM':
        hparams['c_dim'] = 64
        hparams['image_dims'] = [1, 64, 64, 64]
    else:
        raise NotImplementedError

    expt_dir = dir_def.get_expt_dir(hparams)
    hparams['hparams_dir'] = hparams['results_dir'] + 'hparams/' + expt_dir
    hparams['sample_dir'] = hparams['results_dir'] + 'samples/' + expt_dir
    hparams['ckpt_dir'] = hparams['results_dir'] + 'ckpt/' + expt_dir
    hparams['log_dir'] = hparams['results_dir'] + 'logs/' + expt_dir


def setup_dirs(hparams):
    """Setup the dirs"""
    basic_utils.set_up_dir(hparams['hparams_dir'])
    basic_utils.set_up_dir(hparams['ckpt_dir'])
    basic_utils.set_up_dir(hparams['log_dir'])
    basic_utils.set_up_dir(hparams['sample_dir'])
