"""
Some functions here are based on:
https://github.com/carpedm20/DCGAN-tensorflow/blob/master/utils.py

It comes with the following license: https://github.com/carpedm20/DCGAN-tensorflow/blob/master/LICENSE
"""

# pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914, E1101

import glob

import imageio
import numpy as np
import torch
from skimage.transform import resize


def imread(path):
    return imageio.imread(path).astype(np.float)


def center_crop(x, crop_h, crop_w, resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return resize(x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])


def transform(image, input_height, input_width, resize_height=64, resize_width=64, is_crop=True):
    if is_crop:
        cropped_image = center_crop(image, input_height, input_width,
                                    resize_height, resize_width)
    else:
        cropped_image = resize(image, [resize_height, resize_width])
    return torch.from_numpy(cropped_image)/127.5 - 1.


def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, is_crop=True):
    image = imread(image_path)
    image = transform(image, input_height, input_width, resize_height, resize_width, is_crop)
    return image.permute(2, 0, 1).type(torch.FloatTensor)


class CelebAdataset(object):

    def __init__(self, hparams):
        # Get the data file names
        self.datafiles = glob.glob('./data/celebA/*.jpg')
        self.total = len(self.datafiles)

        # Set the pointer to initial location
        self.pos = 0

        # Options for reading the files
        self.input_height = 108
        self.input_width = 108
        self.output_height = hparams['image_dims'][1]
        self.output_width = hparams['image_dims'][2]
        self.is_crop = True

        self.batch_size = hparams['batch_size']
        self.metadata = None

    def __len__(self):
        return len(self.datafiles)

    def __getitem__(self, index):
        return get_image(self.datafiles[index],
                         input_height=self.input_height,
                         input_width=self.input_width,
                         resize_height=self.output_height,
                         resize_width=self.output_width,
                         is_crop=self.is_crop), 0

