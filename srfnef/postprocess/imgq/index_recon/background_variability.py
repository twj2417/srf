import numpy as np

from .contrast import _mean_value, _standard_deviation

"""
Here we have SD,background_variability,SNR,noise
"""


def _mean_values_bg(mask, image):
    num_sphere = np.max(mask)
    values = np.zeros((int(num_sphere - 1), 1))
    for i in range(int(num_sphere) - 1):
        values[i] = _mean_value(mask, image, i + 2)
    return values


def _sd_values_bg(mask, image):
    num_sphere = np.max(mask)
    values = np.zeros((int(num_sphere - 1), 1))
    for i in range(int(num_sphere) - 1):
        values[i] = _standard_deviation(mask, image, i + 2)
    return values


def sd(mask, image):
    value_mean = _mean_values_bg(mask, image)
    return np.std(value_mean, ddof = 1)


def bv(mask, image):
    value_sd = sd(mask, image)
    value_mean = np.mean(_mean_values_bg(mask, image))
    return value_sd / value_mean


def snr2(mask, image):
    u_s = _mean_value(mask, image, 1)
    value_mean = np.mean(_mean_values_bg(mask, image))
    value_sd = noise2(mask, image)
    return (u_s - value_mean) / value_sd


def noise1(mask, image):
    value_mean = _mean_values_bg(mask, image)
    value_sd = _sd_values_bg(mask, image)
    return np.sum(value_sd / value_mean) / value_mean.size


def noise2(mask, image):
    return np.mean(_sd_values_bg(mask, image))
