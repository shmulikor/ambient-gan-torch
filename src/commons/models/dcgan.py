import time as t

import torch
import torch.nn as nn
import torch.optim as optim
from commons.models.gan import GAN_Model, SAVE_PER_TIMES


class DCGAN(GAN_Model):
    def __init__(self, generator, discriminator, dataset, hparams):
        super().__init__(generator, discriminator, dataset, hparams)

        print("initialize DCGAN model")

        # binary cross entropy loss and optimizer
        self.loss = nn.BCELoss()

        # # Values for reference
        # # Using lower learning rate than suggested by (ADAM authors) lr=0.0002  and Beta_1 = 0.5 instead of 0.9 works better [Radford2015]
        # self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        # self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def get_G_optimizer(self):
        learning_rate = self.hparams['g_lr']
        b1 = self.hparams['opt_param1']
        b2 = self.hparams['opt_param2']
        G_optimizer = optim.Adam(self.G.parameters(), lr=learning_rate, betas=(b1, b2))
        return G_optimizer

    def get_D_optimizer(self):
        learning_rate = self.hparams['d_lr']
        b1 = self.hparams['opt_param1']
        b2 = self.hparams['opt_param2']
        D_optimizer = optim.Adam(self.D.parameters(), lr=learning_rate, betas=(b1, b2))
        return D_optimizer

    def train(self):
        # self.calc_fid = self.hparams['measurement_type'] != 'QSM_measurement'

        self.t_begin = t.time()

        real_labels = self.get_torch_variable(torch.ones(self.batch_size))
        fake_labels = self.get_torch_variable(torch.zeros(self.batch_size))

        start_epoch, iters_counter = self.load_model()
        self.G.train()

        for epoch in range(start_epoch, self.hparams['epochs']):

            for i, data in enumerate(self.dataloader):

                real_images = self.get_torch_variable(data[0])
                labels = data[1]
                real_measurements = self.measure_images(real_images, labels)

                if iters_counter == 0:
                    self.save_real_measurements(real_measurements)

                # Train discriminator
                # Compute BCE_Loss using real measuremants
                outputs = self.D(real_measurements)
                d_loss_real = self.loss(outputs.flatten(), real_labels)
                D_y = outputs.mean().item()

                # Compute BCE Loss using fake measuremants
                fake_images = self.generate_images(n_images=self.batch_size)
                fake_measurements = self.measure_images(fake_images, labels)

                outputs = self.D(fake_measurements)
                d_loss_fake = self.loss(outputs.flatten(), fake_labels)
                D_f_G_z1 = outputs.mean().item()

                # Optimize discriminator
                d_loss = d_loss_real + d_loss_fake
                self.D.zero_grad()
                d_loss.backward()
                self.D_optimizer.step()

                # Train generator
                # Compute loss with fake measuremants
                fake_images = self.generate_images(n_images=self.batch_size)
                fake_measurements = self.measure_images(fake_images, labels)

                outputs = self.D(fake_measurements)
                g_loss = self.loss(outputs.flatten(), real_labels)
                D_f_G_z2 = outputs.mean().item()

                # Optimize generator
                self.D.zero_grad()
                self.G.zero_grad()
                g_loss.backward()
                self.G_optimizer.step()

                if i % 10 == 0:
                    print("[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(y): %.4f\tD(f(G(z))): %.4f / %.4f" %
                          (epoch, self.hparams['epochs'], i, len(self.dataloader),
                           d_loss.item(), g_loss.item(), D_y, D_f_G_z1, D_f_G_z2))

                if iters_counter % SAVE_PER_TIMES == 0 or \
                        ((epoch == self.hparams['epochs'] - 1) and (i == len(self.dataloader) - 1)):
                    time = t.time() - self.t_begin
                    print("Total iters: {}".format(iters_counter))
                    print("Time {}".format(time))

                    self.save_model(iters=iters_counter)

                    # save grid of generated images
                    self.save_grid(images=fake_images, iters=iters_counter)

                    # ============ TensorBoard logging ============#
                    # (1) Log the scalar values
                    info = {
                        'train/Loss D': d_loss.data,
                        'train/Loss G': g_loss.data,
                        'train/Loss D Real': d_loss_real.data,
                        'train/Loss D Fake': d_loss_fake.data
                    }

                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value.cpu(), iters_counter)

                    # (2) Log images
                    self.log_images(real_images=real_images, real_measurements=real_measurements, iters=iters_counter)

                    # (3) Log inception score
                    if self.has_inception_model:
                        self.log_inception_score(iter=iters_counter)

                    # (4) Log FID score
                    if self.calc_fid:
                        self.log_fid_score(iters=iters_counter)

                iters_counter += 1

        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))

        # Save the trained parameters
        self.save_model(iters_counter)
