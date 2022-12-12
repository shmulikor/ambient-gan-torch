import os
import time as t

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from commons.models.gan import GAN_Model
from pytorch_fid.fid_score import calculate_fid_given_paths
from torchvision import utils

SAVE_PER_TIMES = 100


class DCGAN_MODEL(GAN_Model):
    def __init__(self, generator, discriminator, dataset, hparams, clean_data=True):
        super().__init__(generator, discriminator, dataset, hparams, clean_data)

        print("initialize DCGAN model")

        # binary cross entropy loss and optimizer
        self.loss = nn.BCELoss()

        # # Values for reference
        # # Using lower learning rate than suggested by (ADAM authors) lr=0.0002  and Beta_1 = 0.5 instead od 0.9 works better [Radford2015]
        # self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        # self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def get_G_optimizer(self):
        learning_rate = 0.0002
        b1 = 0.5
        b2 = 0.999
        G_optimizer = optim.Adam(self.G.parameters(), lr=learning_rate, betas=(b1, b2))
        return G_optimizer

    def get_D_optimizer(self):
        learning_rate = 0.0002
        b1 = 0.5
        b2 = 0.999
        D_optimizer = optim.Adam(self.D.parameters(), lr=learning_rate, betas=(b1, b2))
        return D_optimizer

    def train(self):
        self.t_begin = t.time()

        real_labels = self.get_torch_variable(torch.ones(self.batch_size))
        fake_labels = self.get_torch_variable(torch.zeros(self.batch_size))

        start_epoch, iters_counter = self.load_model()

        for epoch in range(start_epoch, self.hparams['epochs']):

            for i, data in enumerate(self.dataloader, 0):

                real_images = self.get_torch_variable(data[0])
                real_measurements = self.measure_images(real_images)

                # save original images
                if self.hparams['dataset'] == 'QSM' and iters_counter % SAVE_PER_TIMES == 0:
                    # as nifti
                    nib.Nifti1Image(real_measurements[0].cpu().detach().numpy().squeeze(), np.eye(4)).to_filename(
                        os.path.join(self.hparams['sample_dir'], f"real_measurements_iter_{str(iters_counter).zfill(3)}.nii.gz"))

                    # as png
                    img_real = utils.make_grid(torch.cat((real_measurements[:, :, :, :, 16],
                                                           real_measurements[:, :, :, 16, :],
                                                           real_measurements[:, :, 16, :, :])), nrow=8, padding=2, normalize=True, scale_each=True)
                    utils.save_image(img_real,
                                     os.path.join(self.hparams['sample_dir'], f"real_measurements_iter_{str(iters_counter).zfill(3)}.png"))

                # save one example of the measurement
                if self.clean_data and iters_counter == 0:
                    self.save_real_measurements(real_measurements)

                # Train discriminator
                # Compute BCE_Loss using real images
                outputs = self.D(real_measurements)
                d_loss_real = self.loss(outputs.flatten(), real_labels)
                D_x = outputs.mean().item()

                # Compute BCE Loss using fake images
                fake_images = self.generate_images(n_images=self.batch_size)
                fake_measurements = self.measure_images(fake_images)

                # save generated images
                if self.hparams['dataset'] == 'QSM' and iters_counter % SAVE_PER_TIMES == 0:
                    nib.Nifti1Image(fake_images[0].cpu().detach().numpy().squeeze(), np.eye(4)).to_filename(
                        os.path.join(self.hparams['sample_dir'], f"img_generator_iter_{str(iters_counter).zfill(3)}.nii.gz"))

                outputs = self.D(fake_measurements)
                d_loss_fake = self.loss(outputs.flatten(), fake_labels)
                D_G_z1 = outputs.mean().item()

                # Optimize discriminator
                d_loss = d_loss_real + d_loss_fake
                self.D.zero_grad()
                d_loss.backward()
                self.D_optimizer.step()

                # Train generator
                # Compute loss with fake images
                fake_images = self.generate_images(n_images=self.batch_size)
                fake_measurements = self.measure_images(fake_images)

                outputs = self.D(fake_measurements)
                g_loss = self.loss(outputs.flatten(), real_labels)
                D_G_z2 = outputs.mean().item()

                # Optimize generator
                self.D.zero_grad()
                self.G.zero_grad()
                g_loss.backward()
                self.G_optimizer.step()

                if i % 10 == 0:
                    print("[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f" %
                          (epoch, self.hparams['epochs'], i, len(self.dataloader),
                           d_loss.item(), g_loss.item(), D_x, D_G_z1, D_G_z2))

                # TODO - extract logging and saving part of both DCGAN and WGANGP to one function
                if iters_counter % SAVE_PER_TIMES == 0 or \
                        ((epoch == self.hparams['epochs'] - 1) and (i == len(self.dataloader) - 1)):
                    time = t.time() - self.t_begin
                    print("Total iters: {}".format(iters_counter))
                    print("Time {}".format(time))
                    self.save_model(iters_counter)

                    # save grid

                    if self.two_dim_img:
                        self.save_2D_grid(filename=f"img_generator_iter_{str(iters_counter).zfill(3)}")
                    else:
                        self.save_slices_grid(filename=f"img_generator_iter_{str(iters_counter).zfill(3)}")
                        # fake = self.generate_images(n_images=6)
                        # img_fake = utils.make_grid(torch.cat((fake[:, :, :, :, 16],
                        #                                        fake[:, :, :, 16, :],
                        #                                        fake[:, :, 16, :, :])), nrow=6, padding=2, normalize=True, scale_each=True)
                        # utils.save_image(img_fake,
                        #                  os.path.join(self.hparams['sample_dir'], f"img_generator_iter_{str(iters_counter).zfill(3)}.png"))

                    # ============ TensorBoard logging ============#
                    # (1) Log the scalar values
                    info = {
                        'train/Loss D': d_loss.data,
                        'train/Loss G': g_loss.data,
                        'train/Loss D Real': d_loss_real.data,
                        'train/Loss D Fake': d_loss_fake.data
                    }

                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value.cpu(), iters_counter + 1)

                    # (2) Log the images
                    if self.two_dim_img:
                        info = {
                            'real_images': self.log_real_images(real_images),
                            'generated_images': self.log_generated_images()
                        }

                        for tag, images in info.items():
                            self.logger.image_summary(tag, images, iters_counter + 1)

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
