# pylint: disable = C0103, C0111, C0301, R0913, R0903

#from tensorflow.examples.tutorials.mnist import input_data

import os
from torch.utils.data import DataLoader, RandomSampler
import torchvision.datasets as dset
import torchvision.transforms as transforms

root = os.path.join(os.path.join(os.getcwd()), 'data')
if not os.path.exists(root):
    os.makedirs(root)


class RealValIterator(object):

    def __init__(self, hparams):
        trans = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.data = dset.MNIST(root=root, train=True, transform=trans, download=True)
        random_sampler = RandomSampler(self.data, replacement=True, num_samples=10 ** 10)
        dataloader = DataLoader(self.data, batch_size=hparams['batch_size'], num_workers=0, sampler=random_sampler)
        self.data_iterator = iter(dataloader)

        self.model_class = hparams['model_class']

    def next(self):
        data, labels = next(self.data_iterator)
        # x_real, y_real = self.data.train.next_batch(hparams.batch_size)
        # x_real = x_real.reshape(hparams.batch_size, 28, 28, 1)
        if self.model_class == 'conditional':
            return data, labels
        elif self.model_class == 'unconditional':
            return data
        else:
            raise NotImplementedError
