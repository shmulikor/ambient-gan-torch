
import glob
import os

import h5py
import numpy as np
import torch
import nibabel as nib

oj = os.path.join

class CosmosDataset(torch.utils.data.Dataset):

    def __init__(self, subs=[1,2,3,4,5,6,8,9], dipole=False, hparams=None):
        # Get the data filenames
        self.data_root = './data/QSM'
        self.patch_size = 64
        self.subs = np.array(subs).astype(int)

        # self.dipole = dipole
        # self.hparams = hparams
        # assert self.dipole is False or self.hparams is not None

        self.patch_list = self.get_patch_list()

    def __len__(self):
        return len(self.patch_list)
    
    def get_fn(self, sub):
        fn = {
            'mask': 'Sub{0:04d}/cosmos/Sub{0:04d}_mask.nii.gz'.format(sub),
            'cosmos': 'Sub{0:04d}/cosmos/Sub{0:04d}_cosmos.nii.gz'.format(sub),
        }
        return fn

    def get_patch_details(self, sub, stride):
        l = []
        fn = self.get_fn(sub)
        mask = nib.load(oj(self.data_root, fn['mask'])).get_fdata()
        for x in range(0, mask.shape[0]-self.patch_size, stride):
            for y in range(0, mask.shape[1]-self.patch_size, stride):
                for z in range(0, mask.shape[2]-self.patch_size, stride):
                    if mask[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size].sum() > 0.1 * self.patch_size**3:
                        l.append((sub, x, y, z))
        return l

    def get_patch_list(self):
        l = []
        stride = self.patch_size // 4
        for sub in self.subs:
            l.extend(self.get_patch_details(sub=sub, stride=stride))
        return l

    def get_patch(self, idx):
        sub, x, y, z = self.patch_list[idx]
        fn = self.get_fn(sub)
        
        cosmos_patch = nib.load(oj(self.data_root, fn['cosmos'])).get_fdata()
        cosmos_patch = cosmos_patch[x:x+self.patch_size, y:y+self.patch_size, z:z+self.patch_size]
        
        return cosmos_patch

    def __getitem__(self, idx):
        cosmos_patch = self.get_patch(idx)
        return cosmos_patch[np.newaxis, :, :, :].astype('float32'), 0


        # file = self.datafiles[index % len(self.datafiles)]
        # with h5py.File(file, 'r') as hf:
        #     img_size = hf['shape']
        #     while True:
        #         x = torch.randint(0, int(img_size[0] - self.patch_size + 1), (1,)).item()
        #         y = torch.randint(0, int(img_size[1] - self.patch_size + 1), (1,)).item()
        #         z = torch.randint(0, int(img_size[2] - self.patch_size + 1), (1,)).item()

        #         m = hf['mask'][x:x + self.patch_size, y:y + self.patch_size, z:z + self.patch_size]
        #         if m.sum() / np.prod(m.shape) > 0.1:
        #             break

        #     data = hf['cosmos'][x:x + self.patch_size, y:y + self.patch_size, z:z + self.patch_size]

        #     if self.dipole:
        #         path = file.split('/')
        #         path[3] = 'dipole_data'
        #         path[4] = path[4].split('.')[0]
        #         dipole_path = '/'.join(path)
        #         oris = np.random.choice(os.listdir(dipole_path), size=self.hparams['num_orientations'], replace=False)
        #         dipole_paths = [f"{dipole_path}/{ori}/{path[-1]}_{ori}_dipole.npy" for ori in oris]
        #         theta = self.sample_theta(self.hparams['num_orientations'], dipole_paths)

        #         data = torch.unsqueeze(torch.FloatTensor(data), dim=0)
        #         data = self.measure(data, theta)

        #         return data, dipole_paths

        # return data[np.newaxis, :, :, :], self.datafiles[index % len(self.datafiles)]

    def sample_theta(self, n_ori, dipole_paths):
        dk_batch = []

        for ori in range(n_ori):
            dk_batch.append(torch.from_numpy(np.load(dipole_paths[ori])))

        dk_batch = torch.stack(dk_batch)
        return dk_batch

    def measure(self, data, theta):
        measurements = []
        for ori in range(len(theta)):
            _, x_dim, y_dim, z_dim = data.shape
            f = torch.fft.fftn(data, s=theta[ori].shape, norm='ortho')
            d_f = theta[ori] * f
            f_inv_d_f = torch.fft.ifftn(d_f, norm='ortho')[:, :x_dim, :y_dim, :z_dim]
            measurements.append(f_inv_d_f.real)
        return torch.stack(measurements).type(torch.cuda.FloatTensor)

