# code was taken from https://github.com/AKASHKADEL/dcgan-mnist/blob/master/utils.py
# and from https://github.com/AKASHKADEL/dcgan-mnist/blob/master/main.py


from __future__ import division

import math

import numpy as np
import torch
import torch.utils.data
from mnist.inception.model import ResNet18
from scipy.stats import entropy
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms
from torchvision.models.inception import inception_v3


def preds2score(preds, splits=10):
    scores = []
    for i in range(splits):
        part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
        kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
        kl = np.mean(np.sum(kl, 1))
        scores.append(np.exp(kl))
    return np.mean(scores), np.std(scores)


def get_inception_score_grayscale(model_path, imgs, batch_size=32, splits=10):

    net = ResNet18().cuda()
    net.load_state_dict(torch.load(model_path))

    trans = transforms.Compose([transforms.ToPILImage(),
                                transforms.Resize(28),
                                transforms.ToTensor()])

    n_batches = int(math.ceil(float(len(imgs)) / float(batch_size)))
    preds = []
    n_preds = 0

    for i in range(n_batches):
        batch = imgs[(i * batch_size):min((i + 1) * batch_size, len(imgs))]
        batch = [trans(imgs[i].squeeze()) for i in range(len(batch))]
        batch = np.concatenate(batch, 0)
        batch = np.expand_dims(batch, axis=1)
        batch = torch.from_numpy(batch).cuda()

        outputs = net(batch)
        pred = outputs.data.tolist()

        preds.append(pred)
        n_preds += outputs.shape[0]

    preds = np.concatenate(preds, 0)
    preds = np.exp(preds) / np.sum(np.exp(preds), 1, keepdims=True)
    mean_, std_ = preds2score(preds, splits)
    return mean_, std_


def get_inception_score_rgb(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """
        Computes the inception score of the generated images imgs
        imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
        cuda -- whether or not to run on GPU
        batch_size -- batch size for feeding into Inception v3
        splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        if x.shape[1] == 1: # change to 3 channels
            x = torch.cat([x, x, x], dim=1)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)
