# This file is based on https://github.com/AshishBora/ambient-gan/blob/master/src/main.py

# pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914

from __future__ import division

from argparse import ArgumentParser

import torch

from celebA.gen import utils as celebA_utils
from commons import basic_utils
from commons import hparams_def
from commons import utils
from commons.models.dcgan import DCGAN_MODEL
from commons.models.wgan_gradient_penalty import WGAN_GP
from commons.arch import Discriminator32, Discriminator64
from commons.arch import Generator32, Generator64
from mnist.gen import utils as mnist_utils

# from cifar10 import utils as cifar10_utils
# from cifar10 import gan_def as cifar10_gan_def


def main(hparams):

    # Set up some stuff according to hparams
    utils.setup_vals(hparams)
    utils.setup_dirs(hparams)

    # print and save hparams.pkl
    basic_utils.print_hparams(hparams)
    basic_utils.save_hparams(hparams)

    # Print device
    print("Using", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # currently only the AmbientGAN method was implemented, and not the other baselines
    if hparams['train_mode'] != 'ambient':
        raise NotImplementedError

    ###### Get network definitions ######
    # image dimensions - we assume that the input images are squared
    if hparams['image_dims'][-1] == 32:
        generator = Generator32(hparams['c_dim'])
        discriminator = Discriminator32(hparams['c_dim'])
    elif hparams['image_dims'][-1] == 64:
        generator = Generator64(hparams['c_dim'])
        discriminator = Discriminator64(hparams['c_dim'])
    else:
        raise NotImplementedError

    # get data iterator
    if hparams['dataset'] == 'mnist':
        data_iterator = mnist_utils.RealValIterator(hparams)
    elif hparams['dataset'] == 'celebA':
        data_iterator = celebA_utils.RealValIterator(hparams)
    # elif hparams.dataset == 'cifar10':
    #     data_iterator = cifar10_utils.RealValIterator(hparams)
    else:
        raise NotImplementedError

    # Define the connections according to model class and run
    if hparams['model_class'] == 'unconditional':
        # if mdevice.output_type == 'vector':
        #     discriminator = gan_def.discriminator_fc

        # define model
        if hparams['model_type'] == 'dcgan':
            model = DCGAN_MODEL(generator, discriminator, data_iterator, hparams)
        elif hparams['model_type'] == 'wgangp':
            model = WGAN_GP(generator, discriminator, data_iterator, hparams)
        else:
            raise NotImplementedError

        model.train()
    else:
        raise NotImplementedError


if __name__ == '__main__':
    PARSER = ArgumentParser()
    PARSER.add_argument('--hparams', type=str, help='Comma separated list of "name=value" pairs.')
    ARGS = PARSER.parse_args()
    HPARAMS = hparams_def.get_hparams(ARGS)
    main(HPARAMS)
