# This file is based on https://github.com/AshishBora/ambient-gan/blob/master/src/main.py

# pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914

from __future__ import division

from argparse import ArgumentParser

import torch
import os

from celebA.gen import utils as celebA_utils
from commons import basic_utils
from commons import hparams_def
from commons import utils
from commons.models.dcgan import DCGAN
from commons.models.wgan_gp import WGAN_GP
from commons.arch import Generator32, Discriminator32_DCGAN, Discriminator32_WGANGP
from commons.arch import Generator64, Discriminator64_DCGAN, Discriminator64_WGANGP
from commons.arch import Generator64_3D, Discriminator64_3D_DCGAN, Discriminator64_3D_WGANGP
from mnist.gen import utils as mnist_utils
from QSM.gen import utils as QSM_utils

os.environ['CUDA_VISIBLE_DEVICES'] = '5'


def main(hparams):

    # Set up some stuff according to hparams
    utils.setup_vals(hparams)
    utils.setup_dirs(hparams)

    # print and save hparams.pkl
    basic_utils.print_hparams(hparams)
    basic_utils.save_hparams(hparams)

    # print(torch.cuda.is_available())
    
    # print device
    print("Using", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    if hparams['dataset'] == 'mnist':
        dataset = mnist_utils.MNISTdataset()
        generator = Generator32()
        if hparams['model_type'] == 'dcgan':
            discriminator = Discriminator32_DCGAN()
        elif hparams['model_type'] == 'wgangp':
            discriminator = Discriminator32_WGANGP()
        else:
            raise NotImplementedError

    elif hparams['dataset'] == 'celebA':
        dataset = celebA_utils.CelebAdataset(hparams)
        generator = Generator64()
        if hparams['model_type'] == 'dcgan':
            discriminator = Discriminator64_DCGAN()
        elif hparams['model_type'] == 'wgangp':
            discriminator = Discriminator64_WGANGP()
        else:
            raise NotImplementedError
    
    elif hparams['dataset'] == 'QSM_cosmos':
        dataset = QSM_utils.CosmosDataset()
        generator = Generator64_3D()
        if hparams['model_type'] == 'dcgan':
            discriminator = Discriminator64_3D_DCGAN()
        elif hparams['model_type'] == 'wgangp':
            discriminator = Discriminator64_3D_WGANGP()
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError

    print(f"dataset is {hparams['dataset']}, its length is {len(dataset)}")

    # define model
    if hparams['model_type'] == 'dcgan':
        model = DCGAN(generator, discriminator, dataset, hparams)
    elif hparams['model_type'] == 'wgangp':
        model = WGAN_GP(generator, discriminator, dataset, hparams)
    else:
        raise NotImplementedError

    model.train()
    model.evaluate(n_images=64)


if __name__ == '__main__':
    PARSER = ArgumentParser()
    PARSER.add_argument('--hparams', type=str, help='Comma separated list of "name=value" pairs.')
    ARGS = PARSER.parse_args()
    HPARAMS = hparams_def.get_hparams(ARGS)
    main(HPARAMS)
