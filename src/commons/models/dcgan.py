import time as t

import torch
import torch.nn as nn
import torch.optim as optim
from commons.models.gan import GAN_Model

SAVE_PER_TIMES = 100


class DCGAN_MODEL(GAN_Model):
    def __init__(self, generator, discriminator, data_iterator, hparams):
        super().__init__(generator, discriminator, data_iterator, hparams)

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

        # # TODO - Attempt to restore variables from checkpoint
        # init_train_iter = basic_utils.try_restore(hparams, sess, model_saver)
        init_train_iter = -1

        for iter in range(init_train_iter + 1, self.hparams['max_train_iter']):

            real_labels = self.get_torch_variable(torch.ones(self.batch_size))
            fake_labels = self.get_torch_variable(torch.zeros(self.batch_size))

            real_images = self.get_torch_variable(self.data.next())
            real_measurements = self.measure_images(real_images)

            # save one example of the measure
            if iter == init_train_iter + 1:
                self.save_real_measurements(real_measurements)

            # Train discriminator
            # Compute BCE_Loss using real images
            outputs = self.D(real_measurements)
            d_loss_real = self.loss(outputs.flatten(), real_labels)

            # Compute BCE Loss using fake images
            fake_images = self.generate_images(self.batch_size)
            fake_measurements = self.measure_images(fake_images)

            outputs = self.D(fake_measurements)
            d_loss_fake = self.loss(outputs.flatten(), fake_labels)

            # Optimize discriminator
            d_loss = d_loss_real + d_loss_fake
            self.D.zero_grad()
            d_loss.backward()
            self.D_optimizer.step()

            # Train generator
            # Compute loss with fake images
            fake_images = self.generate_images(self.batch_size)
            fake_measurements = self.measure_images(fake_images)

            outputs = self.D(fake_measurements)
            g_loss = self.loss(outputs.flatten(), real_labels)

            # Optimize generator
            self.D.zero_grad()
            self.G.zero_grad()
            g_loss.backward()
            self.G_optimizer.step()

            print(f"Iteration: {iter} / {self.hparams['max_train_iter']}, g_loss: {g_loss}, d_loss: {d_loss}")

            if iter % SAVE_PER_TIMES == 0:
                time = t.time() - self.t_begin
                print("Generator iter: {}".format(iter))
                print("Time {}".format(time))
                self.save_model()

                if self.has_inception_model:
                    self.log_inception_score(iter=iter)

                self.save_grid(iter=iter)

                # ============ TensorBoard logging ============#
                # (1) Log the scalar values
                info = {
                    'Loss D': d_loss.data,
                    'Loss G': g_loss.data,
                    'Loss D Real': d_loss_real.data,
                    'Loss D Fake': d_loss_fake.data

                }

                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value.cpu(), iter + 1)

                # (2) Log the images
                info = {
                    'real_images': self.log_real_images(real_images),
                    'generated_images': self.log_generated_images()
                }

                for tag, images in info.items():
                    self.logger.image_summary(tag, images, iter + 1)

        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))

        # Save the trained parameters
        self.save_model()
