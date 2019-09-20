# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: psf_fit.py
@date: 5/7/2019
@desc:
'''
import numpy as np
import os
import scipy.optimize as opt
import h5py
from srfnef import nef_class
import srfnef as nef
from .fitted_x import FittedX
from .fitted_y import FittedY
from .fitted_z import FittedZ
from .point_source import PointSource
from scipy.signal import savgol_filter

_threshold = 1e-16


@nef_class
class PsfFit:
    half_slice_range: int
    half_patch_range: int

    def __call__(self, pnt_img_path):
        data_x = np.array([])
        data_y = np.array([])
        data_z = np.array([])

        print('fitting psf kernel parameters')
        i = 0
        files = os.listdir(pnt_img_path)
        file = f'img_xy_{i}.hdf5'
        while file in files:
            pnt_img = nef.load(nef.PointSource, pnt_img_path + '/' + file)
            out_x = fitting_psf_x(pnt_img, self.half_slice_range, self.half_patch_range)
            out_y = fitting_psf_y(pnt_img, self.half_slice_range, self.half_patch_range)
            if data_x.size == 0:
                data_x = out_x
                data_y = out_y
            else:
                data_x = np.vstack((data_x, out_x))
                data_y = np.vstack((data_y, out_y))
            i += 1
            file = f'img_xy_{i}.hdf5'

        i = 0
        files = os.listdir(pnt_img_path)
        file = f'img_z_{i}.hdf5'
        while file in files:
            pnt_img = nef.load(nef.PointSource, pnt_img_path + '/' + file)
            out_z = fitting_psf_z(pnt_img, self.half_slice_range, self.half_patch_range)
            if data_z.size == 0:
                data_z = out_z
            else:
                data_z = np.vstack((data_z, out_z))
            i += 1
            file = f'img_z_{i}.hdf5'

        from scipy import signal
        b, a = signal.butter(2, 0.001, 'lowpass')
        data_x[:, 0] = signal.filtfilt(b, a, data_x[:, 0])
        b, a = signal.butter(2, 0.1, 'lowpass')

        data_x[:, 1] = signal.filtfilt(b, a, data_x[:, 1])
        data_x[:, 2] = signal.filtfilt(b, a, data_x[:, 2])
        data_y[:, 1] = signal.filtfilt(b, a, data_y[:, 1])
        data_z[:, 1] = signal.filtfilt(b, a, data_z[:, 1])
        return FittedX(data_x), FittedY(data_y), FittedZ(data_z)


def fitting_psf_x(pnt_img, half_slice_range, half_patch_range):
    px = np.round((pnt_img.pos[0] - pnt_img.center[0] + pnt_img.size[0] / 2 - 0.5) /
                  pnt_img.unit_size[0]).astype(np.int32)
    py = np.round((pnt_img.pos[1] - pnt_img.center[1] + pnt_img.size[1] / 2 - 0.5) /
                  pnt_img.unit_size[1]).astype(np.int32)
    pz = np.round((pnt_img.pos[2] - pnt_img.center[2] + pnt_img.size[2] / 2 - 0.5) /
                  pnt_img.unit_size[2]).astype(np.int32)
    nx, ny, nz = pnt_img.shape
    img_avg = np.average(pnt_img.data[
                         max(px - half_patch_range, 0):
                         min(px + half_patch_range + 1, nx),
                         max(py - half_slice_range, 0):
                         min(py + half_slice_range + 1, ny),
                         max(pz - half_slice_range, 0):
                         min(pz + half_slice_range + 1, nz)],
                         axis = (1, 2))
    x = (np.arange(max(px - half_patch_range, 0),
                   min(px + half_patch_range + 1, nx)) + 0.5) * \
        pnt_img.unit_size[0] + pnt_img.center[0] - pnt_img.size[0] / 2
    ind1 = np.arange(max(px - half_patch_range, 0), px + 1)
    ind2 = np.arange(px, min(px + half_patch_range + 1, nx))

    x1 = (ind1 + 0.5) * pnt_img.unit_size[0] + pnt_img.center[0] - pnt_img.size[0] / 2
    x2 = (ind2 + 0.5) * pnt_img.unit_size[0] + pnt_img.center[0] - pnt_img.size[0] / 2

    out_x1 = _fit_gaussian_1d_fix_mu(img_avg[ind1], x1, mu = pnt_img.pos[0])
    out_x2 = _fit_gaussian_1d_fix_mu(img_avg[ind2], x2, mu = pnt_img.pos[0])
    _dist_ones = _gaussian_1d_mix(1.0, out_x1[1], out_x2[1], pnt_img.pos[0])(x)
    _amp = np.sum(img_avg) / np.sum(_dist_ones)
    out_x = np.array(
        ([_amp, abs(out_x1[1]), abs(out_x2[1]), pnt_img.pos[0]]))
    return out_x


def fitting_psf_y(pnt_img, half_slice_range, half_patch_range):
    px = np.round((pnt_img.pos[0] - pnt_img.center[0] + pnt_img.size[0] / 2 - 0.5) /
                  pnt_img.unit_size[0]).astype(np.int32)
    py = np.round((pnt_img.pos[1] - pnt_img.center[1] + pnt_img.size[1] / 2 - 0.5) /
                  pnt_img.unit_size[1]).astype(np.int32)
    pz = np.round((pnt_img.pos[2] - pnt_img.center[2] + pnt_img.size[2] / 2 - 0.5) /
                  pnt_img.unit_size[2]).astype(np.int32)
    nx, ny, nz = pnt_img.shape

    img_avg = np.average(pnt_img.data[
                         max(px - half_slice_range, 0):
                         min(px + half_slice_range + 1, nx),
                         max(py - half_patch_range, 0):
                         min(py + half_patch_range + 1, ny),
                         max(pz - half_slice_range, 0):
                         min(pz + half_slice_range + 1, nz)],
                         axis = (0, 2))
    y = (np.arange(max(py - half_patch_range, 0),
                   min(py + half_patch_range + 1, ny)) + 0.5) * \
        pnt_img.unit_size[1] + pnt_img.center[1] - pnt_img.size[1] / 2

    out_y = _fit_gaussian_1d(img_avg, y, mu = pnt_img.pos[1])

    out_y[0] = 1.0
    out_y[1] = abs(out_y[1])
    out_y[2] = pnt_img.pos[1]
    return out_y  # the sigma output is squared


def fitting_psf_z(pnt_img, half_slice_range, half_patch_range):
    px = np.round((pnt_img.pos[0] - pnt_img.center[0] + pnt_img.size[0] / 2 - 0.5) /
                  pnt_img.unit_size[0]).astype(np.int32)
    py = np.round((pnt_img.pos[1] - pnt_img.center[1] + pnt_img.size[1] / 2 - 0.5) /
                  pnt_img.unit_size[1]).astype(np.int32)
    pz = np.round((pnt_img.pos[2] - pnt_img.center[2] + pnt_img.size[2] / 2 - 0.5) /
                  pnt_img.unit_size[2]).astype(np.int32)
    nx, ny, nz = pnt_img.shape

    img_avg = np.average(pnt_img.data[
                         max(px - half_slice_range, 0):
                         min(px + half_slice_range + 1, nx),
                         max(py - half_slice_range, 0):
                         min(py + half_slice_range + 1, ny),
                         max(pz - half_patch_range, 0):
                         min(pz + half_patch_range + 1, nz)],
                         axis = (0, 1))
    z = (np.arange(max(pz - half_patch_range, 0),
                   min(pz + half_patch_range + 1, nz)) + 0.5) * \
        pnt_img.unit_size[2] + pnt_img.center[2] - pnt_img.size[2] / 2
    out_z = _fit_gaussian_1d(img_avg, z, mu = pnt_img.pos[2])

    out_z[0] = 1.0
    out_z[1] = abs(out_z[1])
    out_z[2] = pnt_img.pos[2]
    _dist_ones = _gaussian_1d(*out_z)(z)
    _amp = np.sum(img_avg) / np.sum(_dist_ones)
    return np.array([_amp, abs(out_z[1]), abs(out_z[2])])


def _gaussian_1d(amp, sig, mu):
    return lambda x: amp / np.sqrt(2 * np.pi * sig ** 2) * np.exp(-(x - mu) ** 2 / 2 / sig ** 2)


def _gaussian_1d_fix_mu(amp, sig):
    return lambda x, mu: amp / np.sqrt(2 * np.pi * sig ** 2) * np.exp(-(x - mu) ** 2 / 2 / sig ** 2)


def _gaussian_1d_mix(amp, sig0, sig1, mu):
    def mix_1d_gaussian(x):
        data1 = amp / np.sqrt(2 * np.pi * sig0 ** 2) * np.exp(-(x[x <= mu] - mu) ** 2 / 2 / sig0
                                                              ** 2)
        data2 = amp / np.sqrt(2 * np.pi * sig1 ** 2) * np.exp(-(x[x >= mu] - mu) ** 2 / 2 / sig1
                                                              ** 2)
        data_all = np.hstack((data1[:-1], [(data1[-1] + data2[0]) / 2], data2[0:]))
        return data_all

    return mix_1d_gaussian


def _fit_gaussian_1d(data, pos, **kwargs):
    if 'mu' in kwargs.keys():
        kmu = kwargs['mu']
        mu = kmu[0] if isinstance(kmu, tuple) else kmu
    else:
        mu = 0
    if isinstance(pos, tuple):
        pos = pos[0]

    x, data = pos.ravel(), data.ravel()
    x = x[data > _threshold]
    data = data[data > _threshold]

    def _error_function(p):
        return np.ravel(_gaussian_1d(*p)(x) - data)

    if 'initial_guess' in kwargs.keys():
        init = kwargs['initial_guess']
    else:
        init = [np.max(data), 1, mu]
    p = opt.leastsq(_error_function, init)
    return p[0]


def _fit_gaussian_1d_fix_mu(data, pos, mu, **kwargs):
    if isinstance(pos, tuple):
        pos = pos[0]

    x, data = pos.ravel(), data.ravel()
    x = x[data > _threshold]
    data = data[data > _threshold]

    def _error_function(p):
        return np.ravel(_gaussian_1d_fix_mu(*p)(x, mu) - data)

    if 'initial_guess' in kwargs.keys():
        init = kwargs['initial_guess']
    else:
        init = [np.max(data), 1]
    p = opt.leastsq(_error_function, init)
    return p[0]


def _fit_gaussian_1d_mix(data, pos, **kwargs):
    if 'mu' in kwargs.keys():
        kmu = kwargs['mu']
        mu = kmu[0] if isinstance(kmu, tuple) else kmu
    else:
        mu = 0
    if isinstance(pos, tuple):
        pos = pos[0]

    x, data = pos.ravel(), data.ravel()
    x = x[data > _threshold]
    data = data[data > _threshold]

    def _error_function(p):
        return np.ravel(_gaussian_1d_mix(*p)(x) - data)

    if 'initial_guess' in kwargs.keys():
        init = kwargs['initial_guess']
    else:
        init = [np.max(data), 0.5, 1, mu]
    p = opt.leastsq(_error_function, init)
    return p[0]
