# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: image_metric.py
@date: 6/26/2019
@desc:
'''
from srfnef import nef_class, Image
from srfnef.ops.mask.parser import MaskParser
import numpy as np
from functools import reduce


@nef_class
class ContrastHot(MaskParser):
    def __call__(self, mask: np.ndarray, image: Image):
        if len(mask.shape) == 2:
            image_ = image.central_slices[2]
        else:
            image_ = image.data
        mean_back = np.mean(image_[self._get_phantom(mask)])
        return [(ind, np.mean(image_[mk]) / mean_back) for ind, mk in
                self._get_single_hot_source(mask)]


contrast_hot = ContrastHot()


@nef_class
class ContrastCold(MaskParser):
    def __call__(self, mask: np.ndarray, image: Image):
        if len(mask.shape) == 2:
            image_ = image.central_slices[2]
        else:
            image_ = image.data
        mean_back = np.mean(image_[self._get_phantom(mask)])
        return [(ind, np.mean(image_[mk]) / mean_back) for ind, mk in
                self._get_single_cold_source(mask)]


contrast_cold = ContrastCold()


@nef_class
class ContrastNoiseRatio(MaskParser):
    mode: int

    def __call__(self, mask: np.ndarray, image: Image):
        if self.mode == 1:
            if len(mask.shape) == 2:
                image_ = image.central_slices[2]
            else:
                image_ = image.data
            return (np.mean(image_[self._get_hot_source(mask)]) \
                    - np.mean(image_[self._get_phantom(mask)])) \
                   / np.std(image_[self._get_phantom(mask)], ddof = 1)
        elif self.mode == 2:
            if len(mask.shape) == 2:
                image_ = image.central_slices[2]
            else:
                image_ = image.data
            return np.mean(image_[self._get_hot_source(mask)]) \
                   / np.std(image_[self._get_phantom(mask)], ddof = 1)
        else:
            raise NotImplementedError


cnr1 = ContrastNoiseRatio(1)
cnr2 = ContrastNoiseRatio(2)


@nef_class
class ContrastRecoveryCoefficiency(MaskParser):
    mode: int

    def __call__(self, mask: np.ndarray, image: Image):
        if self.mode == 1:
            if len(mask.shape) == 2:
                image_ = image.central_slices[2]
            else:
                image_ = image.data
            return np.max(image_[self._get_hot_source(mask)])
        elif self.mode == 2:
            if len(mask.shape) == 2:
                image_ = image.central_slices[2]
            else:
                image_ = image.data
            return np.mean(image_[self._get_hot_source(mask)]) \
                   / np.mean(image_[self._get_phantom(mask)])

        else:
            raise NotImplementedError


crc1 = ContrastRecoveryCoefficiency(1)
crc2 = ContrastRecoveryCoefficiency(2)


@nef_class
class StandardError(MaskParser):
    def __call__(self, mask: np.ndarray, image: Image):
        if len(mask.shape) == 2:
            image_ = image.central_slices[2]
        else:
            image_ = image.data
        num_bg = 0
        mask_ = np.zeros(image_.shape).astype(np.bool)
        for ind, m in self._get_single_phantom(mask):
            mask_ += np.logical_or(mask_, m)
            num_bg += 1
        return np.std(image_[mask_], ddof = 1) / np.sqrt(num_bg)


standard_error = StandardError()


@nef_class
class NormalizedStandardDeviation(MaskParser):
    def __call__(self, mask: np.ndarray, image: Image):
        if len(mask.shape) == 2:
            image_ = image.central_slices[2]
        else:
            image_ = image.data
        mask_ = self._get_hot_source(mask)
        return np.std(image_[mask_], ddof = 1) / np.mean(image_[mask_])


nsd = NormalizedStandardDeviation()


@nef_class
class StandardDeviation(MaskParser):
    def __call__(self, mask: np.ndarray, image: Image):
        if len(mask.shape) == 2:
            image_ = image.central_slices[2]
        else:
            image_ = image.data
        num_bg = 0
        mask_ = np.zeros(image_.shape).astype(np.bool)
        for ind, m in self._get_single_phantom(mask):
            mask_ += np.logical_or(mask_, m)
            num_bg += 1
        return np.std(image_[mask_], ddof = 1)


sd = StandardDeviation()


@nef_class
class BackgroundVisibility(MaskParser):
    def __call__(self, mask: np.ndarray, image: Image):
        if len(mask.shape) == 2:
            image_ = image.central_slices[2]
        else:
            image_ = image.data
        mask_ = reduce(np.logical_or, [m for _, m in self._get_single_phantom((mask))])
        return np.std(image_[mask_], ddof = 1) / np.mean(image_[mask_])


bg_visibility = BackgroundVisibility()


@nef_class
class Noise(MaskParser):
    mode: int

    def __call__(self, mask: np.ndarray, image: Image):
        if self.mode == 1:
            if len(mask.shape) == 2:
                image_ = image.central_slices[2]
            else:
                image_ = image.data
            value_mean = np.mean(image_[self._get_phantom(mask)])
            mask_ = reduce(np.logical_or, [m for _, m in self._get_single_phantom((mask))])

            return np.sum(np.std(image_[mask_], ddof = 1) / value_mean) / value_mean.size
        elif self.mode == 2:
            if len(mask.shape) == 2:
                image_ = image.central_slices[2]
            else:
                image_ = image.data
            mask_ = reduce(np.logical_or, [m for _, m in self._get_single_phantom((mask))])

            value_sd = np.mean(np.std(image_[mask_], ddof = 1))

            return np.mean(value_sd)
        else:
            raise NotImplementedError


noise1 = Noise(1)
noise2 = Noise(2)


@nef_class
class SignalNoiseRatio(MaskParser):
    mode: int

    def __call__(self, mask: np.ndarray, image: Image):
        if self.mode == 1:
            if len(mask.shape) == 2:
                image_ = image.central_slices[2]
            else:
                image_ = image.data
            mask_ = self._get_hot_source(mask)
            return np.std(image_[mask_], ddof = 1) / np.mean(image_[mask_])
        elif self.mode == 2:
            if len(mask.shape) == 2:
                image_ = image.central_slices[2]
            else:
                image_ = image.data
            u_s = np.mean(image_[self._get_phantom_except(mask)])
            value_mean = np.mean(image_[self._get_phantom(mask)])
            value_sd = Noise(2)(mask, image)
            return (u_s - value_mean) / value_sd


snr1 = SignalNoiseRatio(1)
snr2 = SignalNoiseRatio(2)


@nef_class
class PositiveDeviation(MaskParser):
    def __call__(self, mask: np.ndarray, image: Image):
        if len(mask.shape) == 2:
            image_ = image.central_slices[2]
        else:
            image_ = image.data
        mask_ = reduce(np.logical_or, [m for _, m in self._get_single_phantom((mask))])
        mask2_ = np.logical_not(mask_)
        return 100 * (np.max(image_[mask_]) - np.mean(image_[mask2_])) / np.mean(image_[mask2_])


pos_dev = PositiveDeviation()
