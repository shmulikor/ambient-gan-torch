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
    return np.array(cropped_image)/127.5 - 1.


def get_image(image_path, input_height, input_width, resize_height=64, resize_width=64, is_crop=True):
    image = imread(image_path)
    return transform(image, input_height, input_width, resize_height, resize_width, is_crop)


class RealValIterator(object):

    def __init__(self, hparams):
        # Get the data file names
        self.datafiles = glob.glob('./data/celebA/*.jpg')
        self.total = len(self.datafiles)
        print('Length of data = {}\n'.format(self.total))

        # Set the pointer to initial location
        self.pos = 0

        # Options for reading the files
        self.input_height = 108
        self.input_width = 108
        self.output_height = hparams['image_dims'][1]
        self.output_width = hparams['image_dims'][2]
        self.is_crop = True

        self.batch_size = hparams['batch_size']
        self.model_class = hparams['model_class']

    def __len__(self):
        return self.total

    def next(self):
        start = (self.batch_size * self.pos) % self.total
        stop = self.batch_size * (self.pos + 1) % self.total
        self.pos += 1

        if start < stop:
            batch_files = self.datafiles[start:stop]
        else:
            batch_files = self.datafiles[start:] + self.datafiles[:stop]

        x_real = [get_image(batch_file,
                            input_height=self.input_height,
                            input_width=self.input_width,
                            resize_height=self.output_height,
                            resize_width=self.output_width,
                            is_crop=self.is_crop) for batch_file in batch_files]

        x_real = torch.FloatTensor(np.array(x_real)).permute(0, 3, 1, 2)    # to get NCHW

        if self.model_class == 'unconditional':
            return x_real
        elif self.model_class == 'conditional':
            raise NotImplementedError  # Need labels for celebA
        else:
            raise NotImplementedError
