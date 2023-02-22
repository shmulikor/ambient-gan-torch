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
        self.lambda_term = self.hparams['gp_lambda']

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
        self.t_begin = t.time()

        one = self.get_torch_variable(torch.tensor(1, dtype=torch.float))
        mone = one * -1

        start_epoch, iters_counter = self.load_model()
        self.G.train()

        for epoch in range(start_epoch, self.hparams['epochs']):

            for g_iter, data in enumerate(self.dataloader):

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
                    real_measurements = self.measure_images(real_images, labels)

                    # save one example of the measurement
                    if iters_counter == 0:
                        self.save_real_measurements(real_measurements)

                    d_loss_real = self.D(real_measurements)
                    d_loss_real = d_loss_real.mean()
                    d_loss_real.backward(mone)
                    D_x = d_loss_real.item()

                    # Train with fake images
                    fake_images = self.generate_images(n_images=self.batch_size)
                    fake_measurements = self.measure_images(fake_images, labels)

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

                # Generator update
                for p in self.D.parameters():
                    p.requires_grad = False  # to avoid computation

                self.G.zero_grad()
                # train generator
                # compute loss with fake images
                fake_images = self.generate_images(n_images=self.batch_size)
                fake_measurements = self.measure_images(fake_images, labels)

                g_loss = self.D(fake_measurements)
                g_loss = g_loss.mean()
                g_loss.backward(mone)
                D_G_z2 = g_loss.item()
                self.G_optimizer.step()

                if g_iter % 10 == 0:
                    print("[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f" %
                          (epoch, self.hparams['epochs'], g_iter, len(self.dataloader),
                           d_loss.item(), g_loss.item(), D_x, D_G_z1, D_G_z2))

                # Saving model and sampling images every 1000th generator iterations
                if iters_counter % SAVE_PER_TIMES == 0 or \
                        ((epoch == self.hparams['epochs'] - 1) and (g_iter == len(self.dataloader) - 1)):
                    # Testing
                    time = t.time() - self.t_begin
                    print("Generator iter: {}".format(iters_counter))
                    print("Time {}".format(time))

                    self.save_model(iters=iters_counter)

                    self.save_grid(iters=iters_counter)

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

                    # (2) Log images
                    self.log_images(real_images=real_images, iters=iters_counter)

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
