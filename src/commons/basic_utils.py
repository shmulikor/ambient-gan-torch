# pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914, E1101

from __future__ import division

import os
import shutil
import pickle


def print_hparams(hparams):
    for key in hparams.keys():
        print('{} = {}'.format(key, hparams[key]))
    print('')


def save_hparams(hparams):
    pkl_filepath = hparams['hparams_dir'] + 'hparams.pkl'
    with open(pkl_filepath, 'wb') as f:
        pickle.dump(hparams, f)


def set_up_dir(directory, clean=False):
    if os.path.exists(directory):
        if clean:
            shutil.rmtree(directory)
    else:
        os.makedirs(directory)

