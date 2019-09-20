# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: normalization_correct.py
@date: 5/8/2019
@desc:
'''

from srfnef import nef_class
from srfnef.data import Image, Listmode
from scipy import interpolate
from .amp_x import AmplitudeX
from .amp_z import AmplitudeZ
import numpy as np


@nef_class
class NormalizationCorrect:
    amp_x: AmplitudeX
    amp_z: AmplitudeZ

    def __call__(self, image: Image) -> Image:
        fr = interpolate.interp1d(self.amp_x.ux, self.amp_x.ax, fill_value = 'extrapolate')
        x = np.arange(image.shape[0]) * image.unit_size[0] + image.center[0] - image.size[0] / 2
        y = np.arange(image.shape[1]) * image.unit_size[1] + image.center[1] - image.size[1] / 2
        x1, y1 = np.meshgrid(x, y, indexing = 'ij')
        r = np.sqrt(x1 ** 2 + y1 ** 2)
        mask_r = fr(r)
        mask_r = mask_r / np.mean(mask_r)

        fz = interpolate.interp1d(self.amp_z.uz, self.amp_z.az, fill_value = 'extrapolate')
        z = np.arange(image.shape[2]) * image.unit_size[2] + image.center[2] - image.size[2] / 2
        mask_z = fz(np.abs(z))
        mask_z = mask_z / np.mean(mask_z)

        image_out_data = image.data / mask_z ** 2
        # image_out_data = (image_out_data.transpose() * mask_r).transpose()
        return image.update(data = image_out_data)


@nef_class
class NormalizationAngledCorrect:
    def __call__(self, listmode: Listmode) -> Listmode:
        lx = np.abs(listmode.lors.data[:, 0] - listmode.lors.data[:, 3])
        ly = np.abs(listmode.lors.data[:, 1] - listmode.lors.data[:, 4])
        lz = np.abs(listmode.lors.data[:, 2] - listmode.lors.data[:, 5])
        cos_ = np.sqrt(lx ** 2 + ly ** 2 + lz ** 2) / np.sqrt(lx ** 2 + ly ** 2)
        return listmode * cos_.astype(np.float32)
