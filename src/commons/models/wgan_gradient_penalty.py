import time as t

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch import autograd
from torch.autograd import Variable

plt.switch_backend('agg')

from commons.models.gan import GAN_Model

SAVE_PER_TIMES = 100


class WGAN_GP(GAN_Model):
    def __init__(self, generator, discriminator, data_iterator, hparams):
        super().__init__(generator, discriminator, data_iterator, hparams)
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

        # TODO - Attempt to restore variables from checkpoint

        one = self.get_torch_variable(torch.tensor(1, dtype=torch.float))
        mone = one * -1

        for g_iter in range(self.generator_iters):
            # Requires grad, Generator requires_grad = False
            for p in self.D.parameters():
                p.requires_grad = True

            d_loss_real = 0
            d_loss_fake = 0
            Wasserstein_D = 0
            # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
            for d_iter in range(self.critic_iter):
                self.D.zero_grad()

                real_images = self.data.next()
                # Check for batch to have full batch_size
                if (real_images.shape[0] != self.batch_size):
                    continue

                # Train discriminator
                # WGAN - Training discriminator more iterations than generator
                # Train with real images
                real_images = self.get_torch_variable(real_images)
                real_measurements = self.measure_images(real_images)

                # save one example of the measure
                if g_iter == 0 and d_iter == 0:
                    self.save_real_measurements(real_measurements)

                d_loss_real = self.D(real_measurements)
                d_loss_real = d_loss_real.mean()
                d_loss_real.backward(mone)

                # Train with fake images
                fake_images = self.generate_images(n_images=self.batch_size)
                fake_measurements = self.measure_images(fake_images)

                d_loss_fake = self.D(fake_measurements)
                d_loss_fake = d_loss_fake.mean()
                d_loss_fake.backward(one)

                # Train with gradient penalty
                gradient_penalty = self.calculate_gradient_penalty(real_measurements.data, fake_measurements.data)
                gradient_penalty.backward()

                d_loss = d_loss_fake - d_loss_real + gradient_penalty
                Wasserstein_D = d_loss_real - d_loss_fake
                self.D_optimizer.step()
                print(f'  Discriminator iteration: {d_iter + 1}/{self.critic_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

            # Generator update
            for p in self.D.parameters():
                p.requires_grad = False  # to avoid computation

            self.G.zero_grad()
            # train generator
            # compute loss with fake images
            fake_images = self.generate_images(n_images=self.batch_size)
            fake_measurements = self.measure_images(fake_images)

            g_loss = self.D(fake_measurements)
            g_loss = g_loss.mean()
            g_loss.backward(mone)
            self.G_optimizer.step()
            print(f'Generator iteration: {g_iter}/{self.generator_iters}, g_loss: {g_loss}')
            # Saving model and sampling images every 1000th generator iterations
            if g_iter % SAVE_PER_TIMES == 0:
                # Testing
                time = t.time() - self.t_begin
                print("Generator iter: {}".format(g_iter))
                print("Time {}".format(time))
                self.save_model()

                if self.has_inception_model:
                    self.log_inception_score(iter=g_iter)

                self.save_grid(iter=g_iter)

                # ============ TensorBoard logging ============#
                # (1) Log the scalar values
                info = {
                    'Wasserstein distance': Wasserstein_D.data,
                    'Loss D': d_loss.data,
                    'Loss G': g_loss.data,
                    'Loss D Real': d_loss_real.data,
                    'Loss D Fake': d_loss_fake.data
                }

                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value.cpu(), g_iter + 1)

                # (2) Log the images
                info = {
                    'real_images': self.log_real_images(real_images),
                    'generated_images': self.log_generated_images()
                }

                for tag, images in info.items():
                    self.logger.image_summary(tag, images, g_iter + 1)

        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))

        # Save the trained parameters
        self.save_model()

    def calculate_gradient_penalty(self, real_images, fake_images):
        eta = torch.FloatTensor(self.batch_size,1,1,1).uniform_(0,1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
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
