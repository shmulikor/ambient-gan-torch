import os
import time as t

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable
from torchvision import utils

plt.switch_backend('agg')

from commons.models.gan import GAN_Model

SAVE_PER_TIMES = 100


class WGAN_GP(GAN_Model):
    def __init__(self, generator, discriminator, data_iterator, hparams, clean_data=True):
        super().__init__(generator, discriminator, data_iterator, hparams, clean_data)
        print("initialize WGAN_GradientPenalty model")

        # # Values for reference
        # self.d_optimizer = optim.Adam(self.D.parameters(), lr=1e-4, betas=(0.5, 0.999))
        # self.g_optimizer = optim.Adam(self.G.parameters(), lr=1e-4, betas=(0.5, 0.999))

        self.critic_iter = 5
        self.lambda_term = 10

    def get_G_optimizer(self):
        learning_rate = 1e-4
        b1 = 0.5
        b2 = 0.999
        G_optimizer = optim.Adam(self.G.parameters(), lr=learning_rate, betas=(b1, b2))
        return G_optimizer

    def get_D_optimizer(self):
        learning_rate = 1e-4
        b1 = 0.5
        b2 = 0.999
        D_optimizer = optim.Adam(self.D.parameters(), lr=learning_rate, betas=(b1, b2))
        return D_optimizer

    def train(self):
        self.t_begin = t.time()

        one = self.get_torch_variable(torch.tensor(1, dtype=torch.float))
        mone = one * -1

        start_epoch, iters_counter = self.load_model()
        self.G.train()

        for epoch in range(start_epoch, self.hparams['epochs']):

            for g_iter, data in enumerate(self.dataloader, 0):

                # Requires grad, Generator requires_grad = False
                for p in self.D.parameters():
                    p.requires_grad = True

                d_loss_real = 0
                d_loss_fake = 0
                Wasserstein_D = 0

                # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
                for d_iter in range(self.critic_iter):
                    self.D.zero_grad()

                    # Train discriminator
                    # WGAN - Training discriminator more iterations than generator
                    # Train with real images
                    real_images = self.get_torch_variable(data[0])
                    labels = data[1]
                    if self.hparams['dataset'] == 'QSM_phase':
                        labels = np.array(data[1]).T
                        real_measurements = real_images.type(torch.cuda.FloatTensor)
                    else:
                        real_measurements = self.measure_images(real_images, labels)


                    # save original images
                    if d_iter == 0 and self.hparams['dataset'].startswith('QSM') and iters_counter % SAVE_PER_TIMES == 0:
                        # as nifti
                        obj_to_save = real_measurements[0] if len(real_measurements.shape) == 5 else real_measurements[0, 0]
                        nib.Nifti1Image(obj_to_save.cpu().detach().numpy().squeeze(), np.eye(4)).to_filename(
                            os.path.join(self.hparams['sample_dir'], f"real_measurements_iter_{str(iters_counter).zfill(3)}.nii.gz"))

                        # as png
                        slice_num = np.random.choice(real_measurements.shape[-1])
                        if len(real_measurements.shape) == 5:
                            obj_to_save = torch.cat((real_measurements[:, :, :, :, slice_num],
                                                     real_measurements[:, :, :, slice_num, :],
                                                     real_measurements[:, :, slice_num, :, :]))
                        else:
                            obj_to_save = torch.cat((real_measurements[:, 0, :, :, :, slice_num],
                                                     real_measurements[:, 0, :, :, slice_num, :],
                                                     real_measurements[:, 0, :, slice_num, :, :]))

                        img_real = utils.make_grid(obj_to_save, nrow=8, padding=2, normalize=True, scale_each=True)
                        utils.save_image(img_real,
                                         os.path.join(self.hparams['sample_dir'],
                                                      f"real_measurements_iter_{str(iters_counter).zfill(3)}.png"))

                    # save one example of the measurement
                    if self.clean_data and iters_counter == 0:
                        self.save_real_measurements(real_measurements)

                    if len(real_measurements.shape) == 6:
                        real_measurements = real_measurements.flatten(start_dim=0, end_dim=1)
                    d_loss_real = self.D(real_measurements)
                    d_loss_real = d_loss_real.mean()
                    d_loss_real.backward(mone)
                    D_x = d_loss_real.item()

                    # Train with fake images
                    fake_images = self.generate_images(n_images=self.batch_size)
                    fake_measurements = self.measure_images(fake_images, labels)

                    # save generated images
                    if d_iter == 0 and self.hparams['dataset'].startswith('QSM') and iters_counter % SAVE_PER_TIMES == 0:
                        self.G.eval()
                        obj_to_save = fake_measurements[0] if len(fake_measurements.shape) == 5 else fake_measurements[0, 0]
                        nib.Nifti1Image(obj_to_save.cpu().detach().numpy().squeeze(), np.eye(4)).to_filename(
                            os.path.join(self.hparams['sample_dir'], f"img_generator_iter_{str(iters_counter).zfill(3)}.nii.gz"))
                        self.G.train()

                    if len(fake_measurements.shape) == 6:
                        fake_measurements = fake_measurements.flatten(start_dim=0, end_dim=1)
                    d_loss_fake = self.D(fake_measurements)
                    d_loss_fake = d_loss_fake.mean()
                    d_loss_fake.backward(one)
                    D_G_z1 = d_loss_fake.mean().item()

                    # Train with gradient penalty
                    gradient_penalty = self.calculate_gradient_penalty(real_measurements.data, fake_measurements.data)
                    gradient_penalty.backward()

                    d_loss = d_loss_fake - d_loss_real + gradient_penalty
                    Wasserstein_D = d_loss_real - d_loss_fake
                    self.D_optimizer.step()
                    # print(f'  Discriminator iteration: {d_iter + 1}/{self.critic_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

                # Generator update
                for p in self.D.parameters():
                    p.requires_grad = False  # to avoid computation

                self.G.zero_grad()
                # train generator
                # compute loss with fake images
                fake_images = self.generate_images(n_images=self.batch_size)
                fake_measurements = self.measure_images(fake_images, labels)

                if len(fake_measurements.shape) == 6:
                    fake_measurements = fake_measurements.flatten(start_dim=0, end_dim=1)
                g_loss = self.D(fake_measurements)
                g_loss = g_loss.mean()
                g_loss.backward(mone)
                D_G_z2 = g_loss.item()
                self.G_optimizer.step()

                if g_iter % 10 == 0:
                    print("[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f" %
                          (epoch, self.hparams['epochs'], g_iter, len(self.dataloader),
                           d_loss.item(), g_loss.item(), D_x, D_G_z1, D_G_z2))

                # print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss}')
                # Saving model and sampling images every 1000th generator iterations
                # TODO - extract logging and saving part of both DCGAN and WGANGP to one function
                if iters_counter % SAVE_PER_TIMES == 0 or \
                        ((epoch == self.hparams['epochs'] - 1) and (g_iter == len(self.dataloader) - 1)):
                    # Testing
                    time = t.time() - self.t_begin
                    print("Generator iter: {}".format(iters_counter))
                    print("Time {}".format(time))
                    self.save_model(iters_counter)

                    if self.two_dim_img:
                        self.save_2D_grid(filename=f"img_generator_iter_{str(iters_counter).zfill(3)}")
                    else:
                        self.save_slices_grid(filename=f"img_generator_iter_{str(iters_counter).zfill(3)}")

                    # ============ TensorBoard logging ============#
                    # (1) Log the scalar values
                    info = {
                        'train/Wasserstein distance': Wasserstein_D.data,
                        'train/Loss D': d_loss.data,
                        'train/Loss G': g_loss.data,
                        'train/Loss D Real': d_loss_real.data,
                        'train/Loss D Fake': d_loss_fake.data
                    }

                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value.cpu(), iters_counter + 1)

                    # # (2) Log the images
                    # if self.two_dim_img:
                    #     info = {
                    #         'real_images': self.log_real_images(real_images),
                    #         'generated_images': self.log_generated_images()
                    #     }
                    #
                    #     for tag, images in info.items():
                    #         self.logger.image_summary(tag, images, iters_counter + 1)

                    # (3) Log inception score
                    if self.has_inception_model:
                        self.log_inception_score(iter=iters_counter)

                    # (4) Log fid value
                    if self.calc_fid:
                        self.save_fid_images(real=False)
                        fid_score = self.fid_calculator()
                        print(f"FID score: {fid_score}")
                        info = {'train/FID score': fid_score}
                        for tag, value in info.items():
                            self.logger.scalar_summary(tag, value, iters_counter + 1)

                iters_counter += 1

        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))

        # Save the trained parameters
        self.save_model(iters_counter)

    def calculate_gradient_penalty(self, real_images, fake_images):
        assert real_images.shape == fake_images.shape
        if self.two_dim_img:
            eta = torch.FloatTensor(real_images.shape[0], 1, 1, 1).uniform_(0, 1)
            eta = eta.expand(real_images.shape[0], real_images.size(1), real_images.size(2), real_images.size(3))
        else:
            eta = torch.FloatTensor(real_images.shape[0], 1, 1, 1, 1).uniform_(0, 1)
            eta = eta.expand(real_images.shape[0], real_images.size(1), real_images.size(2), real_images.size(3), real_images.size(4))

        if self.cuda:
            eta = eta.cuda(self.cuda_index)
        else:
            eta = eta

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        if self.cuda:
            interpolated = interpolated.cuda(self.cuda_index)
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty
