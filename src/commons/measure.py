# pylint: disable = C0103, C0111, C0301, R0913, R0903, R0914, E1101

"""Implementations of measurement and unmeasurement"""

from __future__ import division

import copy
import torch.nn.functional as F
import cv2
import numpy as np
import torch
from scipy import signal

from commons import measure_utils


def get_mdevice(hparams):
    if hparams['measurement_type'] == 'drop_independent':
        mdevice = DropIndependent(hparams)
    elif hparams['measurement_type'] == 'drop_row':
        mdevice = DropRow(hparams)
    elif hparams['measurement_type'] == 'drop_col':
        mdevice = DropCol(hparams)
    elif hparams['measurement_type'] == 'drop_rowcol':
        mdevice = DropRowCol(hparams)
    elif hparams['measurement_type'] == 'drop_patch':
        mdevice = DropPatch(hparams)
    elif hparams['measurement_type'] == 'keep_patch':
        mdevice = KeepPatch(hparams)
    elif hparams['measurement_type'] == 'extract_patch':
        mdevice = ExtractPatch(hparams)
    elif hparams['measurement_type'] == 'blur_addnoise':
        mdevice = BlurAddNoise(hparams)
    elif hparams['measurement_type'] == 'pad_rotate_project':
        mdevice = PadRotateProject(hparams)
    elif hparams['measurement_type'] == 'pad_rotate_project_with_theta':
        mdevice = PadRotateProjectWithTheta(hparams)
    elif hparams['measurement_type'] == 'QSM_measurement':
        mdevice = QSM_Measurement(hparams)
    else:
        raise NotImplementedError
    return mdevice


class MeasurementDevice(object):
    """Base class for measurement devices"""

    def __init__(self, hparams):
        # self.measurement_type = hparams.measurement_type
        self.batch_dims = [hparams['batch_size']] + hparams['image_dims']
        self.output_type = None  # indicate whether image or vector

    def sample_theta(self, hparams, labels=None):
        """Abstract Method"""
        # Should return theta_val
        raise NotImplementedError

    def measure(self, hparams, x, theta):
        """Abstract Method"""
        # Tensorflow implementation of measurement. Must be differentiable wrt x.
        # Should return x_measured
        raise NotImplementedError

    def measure_np(self, hparams, x_val, theta_val):
        raise NotImplementedError

    def unmeasure_np(self, hparams, x_measured_val, theta_val):
        """Abstract Method"""
        # Should return x_hat
        raise NotImplementedError

class QSM_Measurement(MeasurementDevice):

    def __init__(self, hparams):
        MeasurementDevice.__init__(self, hparams)
        self.output_type = '3D_image'
        self.alpha = torch.nn.Parameter(torch.ones(1)*4)

    def sample_theta(self, hparams, labels=None):
        batch_size, _, x_dim, y_dim, z_dim = self.batch_dims

        dk_batch = []

        for b_n in range(batch_size):
            dk_batch.append([])

            for ori in range(hparams['num_orientations']):
                path = labels[b_n][ori].split('/')
                path.pop(3) # remove the 'whole'
                path[3] = 'dipole_data'
                # path.append(f"{path[4]}_{path[5]}_dipole.npy")
                path[-1] = f"{path[4]}_{path[5]}_dipole.npy"
                dipole_path = '/'.join(path)

                dk_batch[-1].append(torch.from_numpy(np.load(dipole_path)))

        dk_batch = torch.stack([torch.stack(dk_batch[i]) for i in range(len(dk_batch))])
        return dk_batch

    def measure(self, hparams, x, theta):
        measurements = []
        for b_n in range(len(theta)):
            measurements_by_orientation = []
            for ori in range(len(theta[b_n])):
                _, _, x_dim, y_dim, z_dim = x.shape
                f = torch.fft.fftn(x[b_n], s=theta[b_n][ori].shape, norm='ortho')
                d_f = theta[b_n][ori] * f
                f_inv_d_f = torch.fft.ifftn(d_f, norm='ortho')[:, :x_dim, :y_dim, :z_dim]
                measurements_by_orientation.append(f_inv_d_f.real)
            measurements.append(torch.stack(measurements_by_orientation))
        return torch.stack(measurements).type(torch.cuda.FloatTensor)

class DropDevice(MeasurementDevice):

    def __init__(self, hparams):
        MeasurementDevice.__init__(self, hparams)
        self.output_type = 'image'

    def sample_theta(self, hparams, labels=None):
        """Abstract Method"""
        # Should return theta_val
        raise NotImplementedError

    def measure(self, hparams, x, theta):
        if hparams['dataset'] == 'mnist' or hparams['dataset'] == 'celebA':
            x_measured = torch.multiply(theta, x.mul(0.5).add(0.5)).mul(2).add(-1) # only for mnist and celebA
        else:
            x_measured = torch.multiply(theta, x) # for the 3D case
        return x_measured

    def measure_np(self, hparams, x_val, theta_val):
        x_measured_val = theta_val * x_val
        return x_measured_val

    def unmeasure_np(self, hparams, x_measured_val, theta_val):
        if hparams.unmeasure_type == 'medfilt':
            unmeasure_func = lambda image, mask: signal.medfilt(image)
        elif hparams.unmeasure_type == 'inpaint-telea':
            inpaint_type = cv2.INPAINT_TELEA
            unmeasure_func = measure_utils.get_inpaint_func_opencv(hparams, inpaint_type)
        elif hparams.unmeasure_type == 'inpaint-ns':
            inpaint_type = cv2.INPAINT_NS
            unmeasure_func = measure_utils.get_inpaint_func_opencv(hparams, inpaint_type)
        elif hparams.unmeasure_type == 'inpaint-tv':
            unmeasure_func = measure_utils.get_inpaint_func_tv()
        elif hparams.unmeasure_type == 'blur':
            unmeasure_func = measure_utils.get_blur_func()
        else:
            raise NotImplementedError

        x_unmeasured_val = np.zeros_like(x_measured_val)
        for i in range(x_measured_val.shape[0]):
            x_unmeasured_val[i] = unmeasure_func(x_measured_val[i], theta_val[i])

        return x_unmeasured_val


class DropMaskType1(DropDevice):

    def get_noise_shape(self):
        """Abstract Method"""
        # Should return noise_shape
        raise NotImplementedError

    def sample_theta(self, hparams, labels=None):
        noise_shape = self.get_noise_shape()
        # if hparams['dataset'] == 'QSM':
        #     noise_shape[0] *= hparams['num_orientations']
        repeat_axis = np.ones(len(noise_shape) - 1).astype(int)
        repeat_axis[0] = noise_shape[1]
        mask = torch.Tensor(np.stack([torch.rand(noise_shape[2:]).repeat(tuple(repeat_axis))
                                      for _ in range(noise_shape[0])])) # sample the same mask for all channels
        p = hparams['drop_prob']
        mask = (mask >= p).to(torch.float32) #/ (1 - p)
        # theta = torch.ones(self.batch_dims)
        # theta = theta * mask
        return mask


class DropIndependent(DropMaskType1):

    def get_noise_shape(self):
        noise_shape = copy.deepcopy(self.batch_dims)
        return noise_shape


class DropRow(DropMaskType1):

    def get_noise_shape(self):
        noise_shape = copy.deepcopy(self.batch_dims)
        noise_shape[2] = 1
        return noise_shape


class DropCol(DropMaskType1):

    def get_noise_shape(self):
        noise_shape = copy.deepcopy(self.batch_dims)
        noise_shape[1] = 1
        return noise_shape


class DropRowCol(DropDevice):

    def sample_theta(self, hparams, labels=None):
        drop_row = DropRow(hparams)
        mask1 = drop_row.sample_theta(hparams)
        drop_col = DropCol(hparams)
        mask2 = drop_col.sample_theta(hparams)
        theta_val = mask1 * mask2
        return theta_val


class DropMaskType2(DropDevice):

    def sample_theta(self, hparams, labels=None):
        raise NotImplementedError

    def patch_mask(self, hparams):
        k = hparams['patch_size']
        h, w = hparams['image_dims'][1:3]
        patch_mask = np.ones(self.batch_dims)
        for i in range(hparams['batch_size']):
            x, y = np.random.choice(h-k), np.random.choice(w-k)
            patch_mask[i, :, x:x+k, y:y+k] = 0
        return torch.Tensor(patch_mask)


class DropPatch(DropMaskType2):

    def sample_theta(self, hparams, labels=None):
        return self.patch_mask(hparams)


class KeepPatch(DropMaskType2):

    def sample_theta(self, hparams, labels=None):
        return 1 - self.patch_mask(hparams)


class ExtractPatch(MeasurementDevice):

    def __init__(self, hparams):
        MeasurementDevice.__init__(self, hparams)
        self.output_type = 'image'

    def sample_theta(self, hparams, labels=None):
        k = hparams['patch_size']
        h, w = hparams['image_dims'][1:3]
        theta = np.zeros([hparams['batch_size'], 2])
        for i in range(hparams['batch_size']):
            x, y = np.random.choice(h-k), np.random.choice(w-k)
            theta[i, :] = [x, y]
        return torch.IntTensor(theta)

    def measure(self, hparams, x, theta):
        k = hparams['patch_size']
        patch_list = []
        for t in range(hparams['batch_size']):
            i, j = theta[t, 0], theta[t, 1]
            patch = x[t, :, i:i+k, j:j+k]
            patch = torch.reshape(patch, [1, hparams['image_dims'][0], k, k])
            patch_list.append(patch)
        patches = torch.cat(patch_list, dim=0)
        # TODO(abora): Remove padding by using a custom discriminator
        paddings = measure_utils.get_padding_ep(hparams)
        # x_measured = torch.pad(patches, paddings, "CONSTANT", name='x_measured')
        x_measured = F.pad(patches, paddings, mode='constant', value=-1)
        return x_measured

    def unmeasure_np(self, hparams, x_measured_val, theta_val):
        # How to implement this?
        raise NotImplementedError


class BlurAddNoise(MeasurementDevice):

    def __init__(self, hparams):
        MeasurementDevice.__init__(self, hparams)
        self.output_type = 'image'

    def sample_theta(self, hparams, labels=None):
        theta_val = hparams['additive_noise_std'] * torch.randn(*(self.batch_dims))
        return theta_val

    def measure(self, hparams, x, theta):
        x_blurred = measure_utils.blur(hparams, x)
        x_measured = torch.add(x_blurred, theta)
        return x_measured

    def measure_np(self, hparams, x_val, theta_val):
        x_blurred = measure_utils.blur_np(hparams, x_val)
        x_measured = x_blurred + theta_val
        return x_measured

    def unmeasure_np(self, hparams, x_measured_val, theta_val):
        if hparams.unmeasure_type == 'wiener':
            x_unmeasured_val = measure_utils.wiener_deconv(hparams, x_measured_val)
        else:
            raise NotImplementedError
        return x_unmeasured_val


class PadRotateProjectDevice(MeasurementDevice):

    def __init__(self, hparams):
        MeasurementDevice.__init__(self, hparams)
        self.output_type = 'vector'

    def sample_theta(self, hparams, labels=None):
        theta = (2*np.pi)*np.random.random((hparams['batch_size'], hparams['num_angles'])) - np.pi
        return torch.Tensor(theta)

    def unmeasure_np(self, hparams, x_measured_val, theta_val):
        raise NotImplementedError


class PadRotateProject(PadRotateProjectDevice):

    def measure(self, hparams, x, theta): 
        # TODO - replace with torch implementation
        x_padded = measure_utils.pad(hparams, x)
        x_measured_list = []
        for i in range(hparams['num_angles']):
            angles = theta[:, i]
            x_rotated = measure_utils.rotate(x_padded, angles)
            x_measured = measure_utils.project(hparams, x_rotated)
            x_measured_list.append(x_measured)
        x_measured = tf.concat(x_measured_list, axis=1, name='x_measured')
        return x_measured

    def measure_np(self, hparams, x_val, theta_val):
        raise NotImplementedError

    def unmeasure_np(self, hparams, x_measured_val, theta_val):
        raise NotImplementedError


class PadRotateProjectWithTheta(PadRotateProjectDevice):

    def measure(self, hparams, x, theta):
    # TODO - replace with torch implementation
        x_padded = measure_utils.pad(hparams, x)
        x_measured_list = []
        for i in range(hparams.num_angles):
            angles = theta[:, i]
            x_rotated = measure_utils.rotate(x_padded, angles)
            x_projected = measure_utils.project(hparams, x_rotated)
            x_measured = measure_utils.concat(x_projected, angles)
            x_measured_list.append(x_measured)
        x_measured = tf.concat(x_measured_list, axis=1, name='x_measured')
        return x_measured

    def measure_np(self, hparams, x_val, theta_val):
        raise NotImplementedError

    def unmeasure_np(self, hparams, x_measured_val, theta_val):
        raise NotImplementedError
