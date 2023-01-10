
import glob
import os
from itertools import combinations

import h5py
import numpy as np
import torch
import torchvision.transforms as transforms


# def get_image(batch_file):
#     trans = transforms.Compose([transforms.ToTensor()])
#
#     img = np.load(batch_file)
#     img = trans(img)
#     return torch.unsqueeze(img, dim=0)
#
#
# def get_image_from_whole(batch_file, patch_size=64):
#     nf = np.load(batch_file, 'r')
#     img_size = nf.shape
#     while True:
#         x = torch.randint(0, int(img_size[0] - patch_size + 1), (1,)).item()
#         y = torch.randint(0, int(img_size[1] - patch_size + 1), (1,)).item()
#         z = torch.randint(0, int(img_size[2] - patch_size + 1), (1,)).item()
#
#         # mask = nf != 0
#         m = nf['mask'][x:x + patch_size, y:y + patch_size, z:z + patch_size]
#         if m.sum() / np.prod(m.shape) >= 0.5:
#             break
#
#     data = nf['cosmos'][x:x + patch_size, y:y + patch_size, z:z + patch_size]
#
#     imageout = data
#
#     return imageout


class PhaseDataset(torch.utils.data.Dataset):

    def __init__(self, hparams):
        # self.root = './data/QSM/phase_pdata'
        self.root = './data/QSM/whole/phase_data'
        self.mask_root = './data/QSM/whole/mask_data'
        self.cosmos_root = './data/QSM/whole/cosmos_data'

        self.n_ori = hparams['num_orientations']

        self.datafiles = self.get_data_files()

        self.sub_to_mask = self.load_masks()
        self.sub_to_cosmos = self.load_cosmos()
        self.patch_size = 64

    def load_masks(self):
        all_masks = glob.glob(f"{self.mask_root}/*/*.npy")
        sub_to_mask_file = {mask.split('/')[5]: mask for mask in all_masks}
        sub_to_mask = {k: np.load(v) for k, v in sub_to_mask_file.items()}
        return sub_to_mask

    def load_cosmos(self):
        all_cosmos = glob.glob(f"{self.cosmos_root}/*/*.npy")
        sub_to_cosmos_file = {mask.split('/')[5]: mask for mask in all_cosmos}
        sub_to_cosmos = {k: np.load(v) for k, v in sub_to_cosmos_file.items()}
        return sub_to_cosmos

    def get_data_files(self):
        subs = os.listdir(self.root)
        sub_to_n_phases = {sub: len(os.listdir(f"{self.root}/{sub}/ori1")) for sub in subs}
        sub_to_oris = {sub: os.listdir(os.path.join(self.root, sub)) for sub in subs}
        sub_to_subsets = {sub: list(combinations(oris, self.n_ori))
                               for sub, oris in sub_to_oris.items()}
        data_tuples = [(k, vv) for k, v in sub_to_subsets.items() for vv in v]

        files = []
        for tuple in data_tuples:
            sub, oris = tuple
            for phase in range(sub_to_n_phases[sub]):
                # files.append([f"{self.root}/{sub}/{oris[i]}/{sub}_{oris[i]}_phase_p{phase}.npy"
                #               for i in range(len(oris))])
                files.append([f"{self.root}/{sub}/{oris[i]}/{sub}_{oris[i]}_phase.npy"
                              for i in range(len(oris))])
        return files

    def __len__(self):
        return len(self.datafiles)

    def __getitem__(self, index):
        batch_files = self.datafiles[index]

        # triplet = [self.get_image(file) for file in batch_files]
        triplet = [self.get_image_from_whole(file) for file in batch_files]
        triplet = torch.stack(triplet)

        return triplet, batch_files

    def get_image(self, batch_file):
        trans = transforms.Compose([transforms.ToTensor()])

        img = np.load(batch_file)
        img = trans(img)
        return torch.unsqueeze(img, dim=0)

    def get_image_from_whole(self, batch_file):
        trans = transforms.Compose([transforms.ToTensor()])

        sub = batch_file.split('/')[5]
        nf = np.load(batch_file, 'r')
        img_size = nf.shape
        while True:
            x = torch.randint(0, int(img_size[0] - self.patch_size + 1), (1,)).item()
            y = torch.randint(0, int(img_size[1] - self.patch_size + 1), (1,)).item()
            z = torch.randint(0, int(img_size[2] - self.patch_size + 1), (1,)).item()

            mask = self.sub_to_mask[sub]
            m = mask[x:x + self.patch_size, y:y + self.patch_size, z:z + self.patch_size]
            if m.sum() / np.prod(m.shape) >= 0.5:
                break

        cosmos = self.sub_to_cosmos[sub]
        img = cosmos[x:x + self.patch_size, y:y + self.patch_size, z:z + self.patch_size]
        img = trans(img)

        return torch.unsqueeze(img, dim=0)


class QSMBatchSampler(torch.utils.data.Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through as many times as needed.
    """
    def __init__(self, dataset, batch_size):
        self.primary_indices = [i for i in range(len(dataset.datafiles)) if dataset.datafiles[i][0].split('/')[5] in ['Sub001', 'Sub002', 'Sub003', 'Sub004']]
        self.secondary_indices = [i for i in range(len(dataset.datafiles)) if i not in self.primary_indices]
        self.batch_size = batch_size

    def __iter__(self):
        np.random.shuffle(self.primary_indices)
        np.random.shuffle(self.secondary_indices)
        primary_batches = chunk(self.primary_indices, self.batch_size)
        if len(primary_batches[-1]) != self.batch_size:
            primary_batches = primary_batches[:-1]
        secondary_batches = chunk(self.secondary_indices, self.batch_size)
        if len(secondary_batches[-1]) != self.batch_size:
            secondary_batches = secondary_batches[:-1]
        combined = list(primary_batches + secondary_batches)
        combined = [batch.tolist() for batch in combined]
        np.random.shuffle(combined)
        return iter(combined)

    def __len__(self):
        return (len(self.primary_indices) + len(self.secondary_indices)) // self.batch_size


def chunk(indices, chunk_size):
    return torch.split(torch.tensor(indices), chunk_size)


class CosmosDataset(torch.utils.data.Dataset):

    def __init__(self):
        # Get the data filenames
        self.root = './data/QSM/qsm_h5'
        self.datafiles = glob.glob(f"{self.root}/*.h5", recursive=True)

        self.patch_size = 64

        self.metadata = None

        self.qsm_mean = -0.00024725
        self.qsm_std = 0.02836556

    def __len__(self):
        return len(self.datafiles) * 512

    def __getitem__(self, index):
        with h5py.File(self.datafiles[index % len(self.datafiles)], 'r') as hf:
            img_size = hf['shape']
            while True:
                x = torch.randint(0, int(img_size[0] - self.patch_size + 1), (1,)).item()
                y = torch.randint(0, int(img_size[1] - self.patch_size + 1), (1,)).item()
                z = torch.randint(0, int(img_size[2] - self.patch_size + 1), (1,)).item()

                m = hf['mask'][x:x + self.patch_size, y:y + self.patch_size, z:z + self.patch_size]
                if m.sum() / np.prod(m.shape) >= 0.5:
                    break

            data = hf['cosmos'][x:x + self.patch_size, y:y + self.patch_size, z:z + self.patch_size]

        imageout = data

        # imageout = ((imageout - self.qsm_mean) / self.qsm_std / 2)

        return imageout[np.newaxis, :, :, :], self.datafiles[index % len(self.datafiles)]
