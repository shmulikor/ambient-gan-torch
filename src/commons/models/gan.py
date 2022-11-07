import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
import pickle
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
import os
from itertools import chain
from torchvision import utils

from commons import measure
from commons.inception import get_inception_score_rgb, get_inception_score_grayscale
from commons.logger import Logger
from abc import ABC, abstractmethod


def to_np(x):
    return x.data.cpu().numpy()


class GAN_Model(ABC):
    def __init__(self, generator, discriminator, data_iterator, hparams):
        self.data = data_iterator
        print("initialize GAN model")
        self.G = generator
        self.D = discriminator
        self.C = hparams['c_dim']

        # Check if cuda is available
        self.cuda = False
        self.cuda_index = 0
        self.check_cuda(torch.cuda.is_available())

        self.mdevice = measure.get_mdevice(hparams)
        self.hparams = hparams

        self.batch_size = hparams['batch_size']

        self.G_optimizer = self.get_G_optimizer()
        self.D_optimizer = self.get_D_optimizer()

        # Set the logger
        self.logger = Logger(hparams['log_dir'])
        self.logger.writer.flush()
        self.n_images_to_log = 10

        self.generator_iters = hparams['max_train_iter']

        self.inception_path = f"src/{self.hparams['dataset']}/inception/checkpoints/mnist_model_10.ckpt"
        self.has_inception_model = os.path.isfile(self.inception_path)

    @abstractmethod
    def get_G_optimizer(self):
        pass

    @abstractmethod
    def get_D_optimizer(self):
        pass

    @abstractmethod
    def train(self):
        pass

    def check_cuda(self, cuda_flag=False):
        if cuda_flag:
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            print("Cuda enabled flag: {}".format(self.cuda))

    def get_torch_variable(self, arg):
        if self.cuda:
            return Variable(arg).cuda(self.cuda_index)
        else:
            return Variable(arg)

    def evaluate(self, D_model_path, G_model_path, path_to_save):
        self.load_model(D_model_path, G_model_path)
        samples = self.generate_images(n_images=self.batch_size)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        print(f"Grid of 8x8 images saved to {os.path.join(path_to_save, 'gan_model_image.png')}")
        utils.save_image(grid, os.path.join(path_to_save, 'dgan_model_image.png'))

    def log_real_images(self, images):
        if (self.C == 3):
            return to_np(images.view(-1, self.C, self.hparams['image_dims'][1], self.hparams['image_dims'][2])[:self.n_images_to_log])
        else:
            return to_np(images.view(-1, self.hparams['image_dims'][1], self.hparams['image_dims'][2])[:self.n_images_to_log])

    def log_generated_images(self):
        samples = self.generate_images(n_images=self.n_images_to_log)
        generated_images = []
        for sample in samples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C, self.hparams['image_dims'][1], self.hparams['image_dims'][2]).data.cpu().numpy())
            else:
                generated_images.append(sample.reshape(self.hparams['image_dims'][1], self.hparams['image_dims'][2]).data.cpu().numpy())
        return np.array(generated_images)

    def log_inception_score(self, iter):
        sampled_images = self.generate_images(n_images=800)

        print("Calculating Inception Score over 8k generated images")

        if self.C == 1:
            inception_score = get_inception_score_grayscale(self.inception_path, sampled_images, batch_size=32, splits=10)
        elif self.C == 3:
            inception_score = get_inception_score_rgb(sampled_images, cuda=True, batch_size=32, resize=True, splits=10)
        else:
            raise NotImplementedError

        print("Real Inception score: {}".format(inception_score))
        with open(self.hparams['incpt_pkl'], 'wb') as pickle_file:
            pickle.dump({'inception_score_mean': inception_score[0]}, pickle_file)

        # Log the inception score
        info = {'inception score': inception_score[0]}
        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, iter + 1)

    def save_model(self):
        torch.save(self.G.state_dict(), os.path.join(self.hparams['ckpt_dir'], 'generator.pkl'))
        torch.save(self.D.state_dict(), os.path.join(self.hparams['ckpt_dir'], 'discriminator.pkl'))
        print(f"Models save to {os.path.join(self.hparams['ckpt_dir'], 'generator.pkl')} & "
              f"{os.path.join(self.hparams['ckpt_dir'], 'discriminator.pkl')} ")

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))

    def measure_images(self, images):
        theta = self.get_torch_variable(self.mdevice.sample_theta(self.hparams))
        real_measurements = self.mdevice.measure(self.hparams, images, theta)
        return real_measurements

    def generate_images(self, n_images):
        z = self.get_torch_variable(torch.randn(n_images, 100, 1, 1))
        images = self.G(z)
        return images

    def save_grid(self, iter):
        samples = self.generate_images(n_images=64)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        utils.save_image(grid, os.path.join(self.hparams['sample_dir'], f"img_generator_iter_{str(iter).zfill(3)}.png"))

    def save_real_measurements(self, real_measurements):
        lossy_samples = real_measurements.mul(0.5).add(0.5)
        lossy_samples = lossy_samples.data.cpu()[:self.batch_size]
        grid = utils.make_grid(lossy_samples)
        utils.save_image(grid, os.path.join(self.hparams['sample_dir'],
                                            f"real_measurements.png"))

