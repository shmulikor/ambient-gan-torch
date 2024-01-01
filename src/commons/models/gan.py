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
from torch.utils.data import DataLoader

from commons import measure
from QSM.gen import utils as QSM_utils
from commons.inception import get_inception_score_rgb, get_inception_score_grayscale
from commons.logger import Logger
from abc import ABC, abstractmethod

SAVE_PER_TIMES = 100


# TODO - consider moving those static functions to utils
def to_np(x):
    return x.data.cpu().numpy()

def normalize_image(img):
    return (img - img.min()) / (img.max() - img.min())


class GAN_Model(ABC):
    def __init__(self, generator, discriminator, dataset, hparams):
        # print(len(dataset))
        self.dataloader = DataLoader(dataset, batch_size=hparams['batch_size'], shuffle=True, num_workers=0,
                                     drop_last=True)
        print("initialize GAN model")
        self.G = generator
        self.D = discriminator
        self.channels = hparams['c_dim']

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
        self.two_dim_img = True if len(self.hparams['image_dims']) == 3 else False

        self.use_one_ckpt = self.hparams['use_one_ckpt']

        self.inception_path = f"src/{self.hparams['dataset']}/inception/checkpoints/mnist_model_10.ckpt"
        self.has_inception_model = os.path.isfile(self.inception_path)

        self.calc_fid = True
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
            fake_images = self.generate_images(n_images=n_images)
            self.save_2D_grid(filename=filename, images=fake_images)
            print(f"Grid of images was saved as {os.path.join(self.hparams['sample_dir'], filename)}.png")
        else:
            fid_score = self.save_slices_grid(filename=filename, n_images=n_images, save_3d=True, fid=True)
            print(f"Grid of 2D slices was saved as {os.path.join(self.hparams['sample_dir'], filename)}.png")
            print(f"{n_images} 3D images were saved to {self.hparams['sample_dir']}")
            print(f"The FID score is {fid_score}")

    # def get_images_for_logger(self, images):
    #     if self.two_dim_img:
    #         if self.channels == 3:
    #             return to_np(images.view(-1, self.channels, self.hparams['image_dims'][1], self.hparams['image_dims'][2])[:self.n_images_to_log])
    #         elif self.channels == 1:
    #             return to_np(images.view(-1, self.hparams['image_dims'][1], self.hparams['image_dims'][2])[:self.n_images_to_log])
    #         else:
    #             raise NotImplementedError
    #     else:
    #         return self.get_slices_for_logger(images=images)
    
    # def get_generated_images_for_logger(self):
    #     if self.two_dim_img:
    #         samples = self.generate_images(n_images=self.n_images_to_log)
    #         generated_images = []
    #         for sample in samples:
    #             if self.channels == 3:
    #                 generated_images.append(sample.reshape(self.channels, self.hparams['image_dims'][1], self.hparams['image_dims'][2]).data.cpu().numpy())
    #             elif self.channels == 1:
    #                 generated_images.append(sample.reshape(self.hparams['image_dims'][1], self.hparams['image_dims'][2]).data.cpu().numpy())
    #             else:
    #                 raise NotImplementedError
    #         return np.array(generated_images)
    #     else:
    #         return self.get_slices_for_logger(images=None)

    # def get_slices_for_logger(self, images=None):
    #     n_images = self.n_images_to_log // 3
    #     images = self.generate_images(n_images=n_images) if images is None else images[:n_images]
    #     slice_num = np.random.choice(images.shape[-1])
    #     all_slices = torch.cat((images[:, :, :, :, slice_num], images[:, :, :, slice_num, :], images[:, :, slice_num, :, :]))
    #     normalized = torch.stack([normalize_image(slice) for slice in all_slices])
    #     normalized = 2 * normalized - 1
    #     return to_np(normalized)

    def log_inception_score(self, iter):
        sampled_images = self.generate_images(n_images=800)
        print("Calculating Inception Score over 8k generated images")
        if self.channels == 1:
            inception_score = get_inception_score_grayscale(self.inception_path, sampled_images, batch_size=32, splits=10)
        elif self.channels == 3:
            inception_score = get_inception_score_rgb(sampled_images, cuda=True, batch_size=32, resize=True, splits=10)
        else:
            raise NotImplementedError

        # Log the inception score
        print("Inception score: {}".format(inception_score[0]))
        info = {'score/inception score': inception_score[0]}
        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, iter)

    def save_model(self, iters):
        G_state = {'state_dict': self.G.state_dict(), 'optimizer': self.G_optimizer.state_dict(), 'iters': iters}
        D_state = {'state_dict': self.D.state_dict(), 'optimizer': self.D_optimizer.state_dict(), 'iters': iters}

        G_dir = self.hparams['ckpt_dir']
        if not self.use_one_ckpt:
            G_dir = os.path.join(G_dir, 'generator')
        os.makedirs(G_dir, exist_ok=True)

        D_dir = self.hparams['ckpt_dir']
        if not self.use_one_ckpt:
            D_dir = os.path.join(D_dir, 'discriminator')
        os.makedirs(D_dir, exist_ok=True)

        if self.use_one_ckpt:
            G_path = os.path.join(G_dir, f"generator.pkl")
            D_path = os.path.join(G_dir, f"discriminator.pkl")
        else:
            G_path = os.path.join(G_dir, f"generator_iter_{str(iters).zfill(3)}.pkl")
            D_path = os.path.join(D_dir, f"discriminator_iter_{str(iters).zfill(3)}.pkl")

        torch.save(G_state, G_path)
        torch.save(D_state, D_path)
        print(f"Models were saved to {G_path} & {D_path}")

    def load_model(self, n_iters=-1):
        if not self.hparams['use_saved_model']:
            return 0, 0

        G_path = os.path.join(self.hparams['ckpt_dir'], 'generator.pkl')
        D_path = os.path.join(self.hparams['ckpt_dir'], 'discriminator.pkl')

        iters_counter, start_epoch = 0, 0

        if not self.use_one_ckpt:
            G_dir = os.path.join(self.hparams['ckpt_dir'], 'generator')
            G_models = os.listdir(G_dir)

            D_dir = os.path.join(self.hparams['ckpt_dir'], 'discriminator')
            D_models = os.listdir(D_dir)

            if len(G_models) and len(D_models):
                G_all_models = sorted(G_models, key=lambda x: int(x.split('.')[0].split('_')[-1]))
                G_model = G_all_models[-1] if n_iters == -1 else f"generator_iter_{str(n_iters).zfill(3)}.pkl"
                G_path = os.path.join(G_dir, G_model)

                D_all_models = sorted(D_models, key=lambda x: int(x.split('.')[0].split('_')[-1]))
                D_model = D_all_models[-1] if n_iters == -1 else f"discriminator_iter_{str(n_iters).zfill(3)}.pkl"
                D_path = os.path.join(D_dir, D_model)

                iters_counter = int(G_model.split('.')[0].split('_')[-1])
                start_epoch = iters_counter // len(self.dataloader)

        if not os.path.isfile(G_path) or not os.path.isfile(D_path):
            print('No model to load')
            return start_epoch, iters_counter

        G_state = torch.load(G_path)
        self.G.load_state_dict(G_state['state_dict'])
        self.G_optimizer.load_state_dict(G_state['optimizer'])

        D_state = torch.load(D_path)
        self.D.load_state_dict(D_state['state_dict'])
        self.D_optimizer.load_state_dict(D_state['optimizer'])

        if 'iters' in G_state.keys():
            iters_counter = G_state['iters']
            start_epoch = iters_counter // len(self.dataloader)

        print('Generator model loaded from {}'.format(G_path))
        print('Discriminator model loaded from {}'.format(D_path))

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

    def get_2D_grid(self, images):
        # samples = self.generate_images(n_images=n_images)
        if self.hparams['dataset'] == 'mnist' or self.hparams['dataset'] == 'celebA':
            images = images.mul(0.5).add(0.5)
        images = images.data.cpu()
        grid = utils.make_grid(images)
        return grid

    def save_2D_grid(self, filename, images):
        grid = self.get_2D_grid(images=images)
        utils.save_image(grid, os.path.join(self.hparams['sample_dir'], f"{filename}.png"))

    def save_3D_image(self, image, path):
        nib.Nifti1Image(image.cpu().detach().numpy().squeeze(), np.eye(4)).to_filename(path)
    
    def get_slices_grid(self, images, n_images=8, fid=False):
        self.G.eval()
        fake_images = self.generate_images(n_images=n_images) if not fid else self.save_fid_images(real=False, n_images=n_images)
        self.G.train()
        # if save_3d:
        #     for i in range(n_images):
        #         self.save_3D_image(image=fake_images[i], path=os.path.join(self.hparams['sample_dir'], f"{str(i).zfill(3)}.nii.gz"))
        slice_num = np.random.choice(fake_images.shape[-1])
        slices_grid = utils.make_grid(torch.cat((fake_images[:, :, :, :, slice_num],
                                                 fake_images[:, :, :, slice_num, :],
                                                 fake_images[:, :, slice_num, :, :])),
                                                 nrow=n_images, padding=2, normalize=True, scale_each=True)
        return slices_grid

    def save_slices_grid(self, filename, n_images=8, save_3d=False, fid=False):
        slices_grid = self.get_slices_grid()
        utils.save_image(slices_grid, os.path.join(self.hparams['sample_dir'], f"{filename}.png"))
        # if fid:
        #     return self.fid_calculator()

    def save_real_measurements(self, real_measurements):
        if self.two_dim_img:
            lossy_samples = real_measurements.mul(0.5).add(0.5)
            lossy_samples = lossy_samples.data.cpu()[:self.batch_size]
            grid = utils.make_grid(lossy_samples)
            utils.save_image(grid, os.path.join(self.hparams['sample_dir'],
                                                f"real_measurements.png"))
        else:
            real_dir = os.path.join(self.hparams['sample_dir'], 'real_measurements')
            os.makedirs(real_dir, exist_ok=True)

            # save 2D slices
            slice = np.random.choice(real_measurements.shape[-1])
            grid = utils.make_grid(torch.cat((real_measurements[:, :, :, :, slice],
                                                  real_measurements[:, :, :, slice, :],
                                                  real_measurements[:, :, slice, :, :])),
                                       nrow=16, padding=2, normalize=True, scale_each=True)
            utils.save_image(grid, os.path.join(real_dir, 'real_measurements_slices.png'))

            # save 3D images
            for i, measurement in enumerate(real_measurements):
                self.save_3D_image(image=measurement, path=os.path.join(real_dir, f"real_measurements_{str(i).zfill(3)}.nii.gz"))

    def path_for_fid(self):
        path = f"src/fid/{self.hparams['dataset']}"
        print(path)
        os.makedirs(path, exist_ok=True)
        os.makedirs(os.path.join(path, 'real'), exist_ok=True)
        os.makedirs(os.path.join(path, 'fake'), exist_ok=True)
        return path

    def save_fid_images(self, real=False, n_images=64):
        self.G.eval()
        path = os.path.join(self.fid_path, 'real' if real else 'fake')
        if real and len(os.listdir(path)):
            return

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

    def save_grid(self, images, iters):
        filename=f"img_generator_iter_{str(iters).zfill(3)}"
        if self.two_dim_img:
            grid = self.get_2D_grid(images=images)
            # utils.save_image(grid, os.path.join(self.hparams['sample_dir'], f"{filename}.png"))
            # self.save_2D_grid(filename=f"img_generator_iter_{str(iters).zfill(3)}")
        else:
            grid = self.get_slices_grid(images=images)
            # self.save_slices_grid(filename=f"img_generator_iter_{str(iters).zfill(3)}")
        utils.save_image(grid, os.path.join(self.hparams['sample_dir'], f"{filename}.png"))

    def log_images(self, real_images, real_measurements, iters):
        if self.two_dim_img:
            self.G.eval()
            fake_images = self.generate_images(n_images=64)
            self.G.train()
            
            info = {'images/generated_images': self.get_2D_grid(images=fake_images)}
            if iters == SAVE_PER_TIMES:
                info_real = {
                    'images/real_images': self.get_2D_grid(images=real_images),
                    'images/measured_images': self.get_2D_grid(images=real_measurements)
                }
                info.update(info_real)

            for tag, images in info.items():
                self.logger.image_summary(tag, images, iters)

    def log_fid_score(self, iters):
        self.save_fid_images(real=False)
        fid_score = self.fid_calculator()
        print(f"FID score: {fid_score}")
        info = {'score/FID score': fid_score}
        for tag, value in info.items():
            self.logger.scalar_summary(tag, value, iters)
