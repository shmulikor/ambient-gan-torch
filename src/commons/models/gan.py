import pickle

import matplotlib.pyplot as plt
import torch
from pytorch_fid.fid_score import calculate_fid_given_paths
from torch.autograd import Variable

plt.switch_backend('agg')
import numpy as np
import os
import nibabel as nib
from torchvision import utils

from commons import measure
from QSM.gen import utils as QSM_utils
from commons.inception import get_inception_score_rgb, get_inception_score_grayscale
from commons.logger import Logger
from abc import ABC, abstractmethod


# TODO - consider moving those static functions to utils
def to_np(x):
    return x.data.cpu().numpy()

def normalize_image(img):
    return (img - img.min()) / (img.max() - img.min())


class GAN_Model(ABC):
    def __init__(self, generator, discriminator, dataset, hparams, clean_data=True):
        if hparams['dataset'] == 'QSM_phase':
            batch_sampler = QSM_utils.QSMBatchSampler(dataset, batch_size=hparams['batch_size'])
            self.dataloader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_sampler=batch_sampler)
        else:
            self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=hparams['batch_size'], shuffle=True,
                                                          num_workers=0, drop_last=True)
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
        self.clean_data = clean_data

        self.batch_size = hparams['batch_size']

        self.G_optimizer = self.get_G_optimizer()
        self.D_optimizer = self.get_D_optimizer()

        # Set the logger
        self.logger = Logger(hparams['log_dir'])
        self.logger.writer.flush()
        self.n_images_to_log = 10
        self.two_dim_img = True if len(self.hparams['image_dims']) == 3 else False

        # self.generator_iters = hparams['max_train_iter']

        self.inception_path = f"src/{self.hparams['dataset']}/inception/checkpoints/mnist_model_10.ckpt"
        self.has_inception_model = os.path.isfile(self.inception_path)

        # self.calc_fid = self.hparams['dataset'] == 'QSM' # TODO - generalize to the 2D case
        self.calc_fid = True
        if self.calc_fid:
            self.fid_path = self.path_for_fid()
            self.save_fid_images(real=True)

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

    def evaluate(self, n_iters=-1, filename='model_grid', n_images=64):
        self.load_model(n_iters=n_iters)
        if self.two_dim_img:
            self.save_2D_grid(filename=filename, n_images=n_images)
            print(f"Grid of images was saved as {os.path.join(self.hparams['sample_dir'], filename)}.png")
        else:
            fid_score = self.save_slices_grid(filename=filename, n_images=n_images, save_3d=True, fid=True)
            print(f"Grid of 2D slices was saved as {os.path.join(self.hparams['sample_dir'], filename)}.png")
            print(f"{n_images} 3D images were saved to {self.hparams['sample_dir']}")
            print(f"The FID score is {fid_score}")


    def log_real_images(self, images):
        if self.two_dim_img:
            if self.C == 3:
                return to_np(images.view(-1, self.C, self.hparams['image_dims'][1], self.hparams['image_dims'][2])[:self.n_images_to_log])
            else:
                return to_np(images.view(-1, self.hparams['image_dims'][1], self.hparams['image_dims'][2])[:self.n_images_to_log])
        else:
            return self.log_slices(images=images)

    def log_generated_images(self):
        if self.two_dim_img:
            samples = self.generate_images(n_images=self.n_images_to_log)
            generated_images = []
            for sample in samples:
                if self.C == 3:
                    generated_images.append(sample.reshape(self.C, self.hparams['image_dims'][1], self.hparams['image_dims'][2]).data.cpu().numpy())
                else:
                    generated_images.append(sample.reshape(self.hparams['image_dims'][1], self.hparams['image_dims'][2]).data.cpu().numpy())
            return np.array(generated_images)
        else:
            return self.log_slices(images=None)


    def log_slices(self, images=None):
        n_images = self.n_images_to_log // 3
        images = self.generate_images(n_images=n_images) if images is None else images[:n_images]
        slice_num = np.random.choice(images.shape[-1])
        all_slices = torch.cat((images[:, :, :, :, slice_num], images[:, :, :, slice_num, :], images[:, :, slice_num, :, :]))
        normalized = torch.stack([normalize_image(slice) for slice in all_slices])
        normalized = 2 * normalized - 1
        return to_np(normalized)


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

    def save_model(self, iters):
        G_state = {'state_dict': self.G.state_dict(), 'optimizer': self.G_optimizer.state_dict()}
        D_state = {'state_dict': self.D.state_dict(), 'optimizer': self.D_optimizer.state_dict()}

        G_dir = os.path.join(self.hparams['ckpt_dir'], 'generator')
        if not os.path.isdir(G_dir):
            os.makedirs(G_dir)

        D_dir = os.path.join(self.hparams['ckpt_dir'], 'discriminator')
        if not os.path.isdir(D_dir):
            os.makedirs(D_dir)

        G_path = os.path.join(G_dir, f"generator_iter_{str(iters).zfill(3)}.pkl")
        D_path = os.path.join(D_dir, f"discriminator_iter_{str(iters).zfill(3)}.pkl")
        torch.save(G_state, G_path)
        torch.save(D_state, D_path)
        print(f"Models were saved to {G_path} & {D_path}")

    def load_model(self, n_iters=-1):
        G_dir = os.path.join(self.hparams['ckpt_dir'], 'generator')
        if not os.path.isdir(G_dir):
            os.makedirs(G_dir)
        G_models = os.listdir(G_dir)

        D_dir = os.path.join(self.hparams['ckpt_dir'], 'discriminator')
        if not os.path.isdir(D_dir):
            os.makedirs(D_dir)
        D_models = os.listdir(D_dir)

        if len(G_models) and len(D_models):
            G_all_models = sorted(G_models, key=lambda x: int(x.split('.')[0].split('_')[-1]))
            D_all_models = sorted(D_models, key=lambda x: int(x.split('.')[0].split('_')[-1]))

            G_model = G_all_models[-1] if n_iters == -1 else f"generator_iter_{str(n_iters).zfill(3)}.pkl"
            D_model = D_all_models[-1] if n_iters == -1 else f"discriminator_iter_{str(n_iters).zfill(3)}.pkl"

            G_path = os.path.join(G_dir, G_model)
            G_state = torch.load(G_path)
            self.G.load_state_dict(G_state['state_dict'])
            self.G_optimizer.load_state_dict(G_state['optimizer'])

            D_path = os.path.join(D_dir, D_model)
            D_state = torch.load(D_path)
            self.D.load_state_dict(D_state['state_dict'])
            self.D_optimizer.load_state_dict(D_state['optimizer'])

            iters_counter = int(G_model.split('.')[0].split('_')[-1])
            start_epoch = iters_counter // len(self.dataloader)
            print('Generator model loaded from {}'.format(G_path))
            print('Discriminator model loaded from {}'.format(D_path))
        else:
            iters_counter = 0
            start_epoch = 0
            print('no model to load')
        return start_epoch, iters_counter

    def measure_images(self, images, labels=None):
        theta = self.get_torch_variable(self.mdevice.sample_theta(self.hparams, labels))
        measurements = self.mdevice.measure(self.hparams, images, theta)
        return measurements

    def generate_images(self, n_images):
        if self.two_dim_img:
            z = self.get_torch_variable(torch.randn(n_images, self.hparams['z_dim'], 1, 1))
        else:
            z = self.get_torch_variable(torch.randn(n_images, self.hparams['z_dim'], 1, 1, 1))
        images = self.G(z)
        return images

    def save_2D_grid(self, filename, n_images=64):
        samples = self.generate_images(n_images=n_images)
        if self.hparams['dataset'] == 'mnist' or self.hparams['dataset'] == 'celebA':
            samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        utils.save_image(grid, os.path.join(self.hparams['sample_dir'], f"{filename}.png"))

    def save_3D_image(self, image, filename):
        nib.Nifti1Image(image.cpu().detach().numpy().squeeze(), np.eye(4)).to_filename(
            os.path.join(self.hparams['sample_dir'], f"{filename}.nii.gz"))


    def save_slices_grid(self, filename, n_images=8, save_3d=False, fid=False):
        self.G.eval()
        fake_images = self.generate_images(n_images=n_images) if not fid else self.save_fid_images(real=False, n_images=n_images)
        if save_3d:
            for i in range(n_images):
                self.save_3D_image(image=fake_images[i], filename=str(i).zfill(3))
        slice_num = np.random.choice(fake_images.shape[-1])
        img_fake = utils.make_grid(torch.cat((fake_images[:, :, :, :, slice_num],
                                              fake_images[:, :, :, slice_num, :],
                                              fake_images[:, :, slice_num, :, :])),
                                   nrow=n_images, padding=2, normalize=True, scale_each=True)
        utils.save_image(img_fake, os.path.join(self.hparams['sample_dir'], f"{filename}.png"))
        self.G.train()
        if fid:
            return self.fid_calculator()

    def save_real_measurements(self, real_measurements):
        if self.two_dim_img:
            lossy_samples = real_measurements.mul(0.5).add(0.5)
            lossy_samples = lossy_samples.data.cpu()[:self.batch_size]
            grid = utils.make_grid(lossy_samples)
            utils.save_image(grid, os.path.join(self.hparams['sample_dir'],
                                                f"real_measurements.png"))
        else:
            real_dir = os.path.join(self.hparams['sample_dir'], 'real_measurements')
            if not os.path.isdir(real_dir):
                os.makedirs(real_dir)
            slice = np.random.choice(real_measurements.shape[-1])
            img_fake = utils.make_grid(torch.cat((real_measurements[:, :, :, :, slice],
                                                  real_measurements[:, :, :, slice, :],
                                                  real_measurements[:, :, slice, :, :])),
                                       nrow=real_measurements.shape[-1], padding=2, normalize=True, scale_each=True)
            utils.save_image(img_fake, real_dir)
            for i, measurement in enumerate(real_measurements):
                self.save_3D_image(image=measurement, filename=os.path.join(real_dir, f"real_measurements_{str(i).zfill(3)}"))

    def path_for_fid(self):
        # path = f"src/fid/{self.hparams['dataset']}"
        path = f"src/fid{self.hparams['dataset']}" if 'src' not in os.getcwd() else f"fid/{self.hparams['dataset']}"
        # TODO - the line above is only for the weird case of running from 2 different base dirs. Don't commit it.
        if not os.path.isdir(path):
            os.makedirs(path)
        if not os.path.isdir(os.path.join(path, 'real')):
            os.makedirs(os.path.join(path, 'real'))
        if not os.path.isdir(os.path.join(path, 'fake')):
            os.makedirs(os.path.join(path, 'fake'))
        return path

    def save_fid_images(self, real=False, n_images=64):
        self.G.eval()
        path = os.path.join(self.fid_path, 'real' if real else 'fake')
        if real and len(os.listdir(path)):
            return
        if real and not self.clean_data:
            print("No clean data for QSM Phase data. Can't save real images for FID score calculation.")
            return
        # TODO - how many real images do I need to save?
        images = next(iter(self.dataloader))[0] if real else self.generate_images(n_images=n_images)
        if self.two_dim_img:
            [utils.save_image(normalize_image(images[i]), f"{path}/{i}.png") for i in range(len(images))]
        else:
            if len(images.shape) == 6:
                images = images.flatten(start_dim=0, end_dim=1)
            val = np.random.choice(self.hparams['image_dims'][-1])
            all_slices = torch.cat((images[:, :, :, :, val], images[:, :, :, val, :], images[:, :, val, :, :]))
            [utils.save_image(normalize_image(all_slices[i]), f"{path}/{i}.png") for i in range(len(all_slices))]
        self.G.train()
        return images

    def fid_calculator(self):
        fid_score = calculate_fid_given_paths(
            paths=[os.path.join(self.fid_path, 'real'), os.path.join(self.fid_path, 'fake')],
            batch_size=self.batch_size, device='cuda' if self.cuda else 'cpu', dims=2048)
        return fid_score