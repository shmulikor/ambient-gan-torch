# pylint: disable = C0103, C0111, C0301, R0913, R0903

#from tensorflow.examples.tutorials.mnist import input_data

import os

import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

root = os.path.join(os.path.join(os.getcwd()), 'data')
if not os.path.exists(root):
    os.makedirs(root)


class MNISTdataset(torch.utils.data.Dataset):

    def __init__(self):
        trans = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.data = dset.MNIST(root=root, train=True, transform=trans, download=True)

        self.metadata = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
