import numpy as np
from scipy.optimize import curve_fit

from .contrast import _mean_value, _standard_deviation

"""
Here we have spatial resolution,sd_rc
"""


def _gaussian(x, *param):
    return param[0] * np.exp(-np.power(x - param[1], 2) / (2 * np.power(param[2], 2)))


def _gaussian_fit_x(image):
    center = np.where(image == np.max(image))
    center = [c[0] for c in center]
    x = np.arange(50)
    y = image[center[0], center[1] - 25:center[1] + 25, center[2]]
    return curve_fit(_gaussian, x, y, p0 = [1, center, 5])


def _gaussian_fit_y(image):
    center = np.where(image == np.max(image))
    center = [c[0] for c in center]
    x = np.arange(50)
    y = image[center[0] - 25:center[0] + 25, center[1], center[2]]
    return curve_fit(_gaussian, x, y, p0 = [1, center, 5])


def _gaussian_fit_z(image):
    center = np.where(image == np.max(image))
    center = [c[0] for c in center]
    x = np.arange(50)
    y = image[center[0], center[1], center[2] - 25:center[2] + 25]
    return curve_fit(_gaussian, x, y, p0 = [1, center, 5])


def fwhm_x(image):
    popt, pcov = _gaussian_fit_x(image.data)
    return 2.35 * popt[2] * image.unit_size[0]


def fwtm_x(image):
    popt, pcov = _gaussian_fit_x(image.data)
    return 4.29 * popt[2] * image.unit_size[0]


def fwhm_y(image):
    popt, pcov = _gaussian_fit_y(image.data)
    return 2.35 * popt[2] * image.unit_size[1]


def fwtm_y(image):
    popt, pcov = _gaussian_fit_y(image.data)
    return 4.29 * popt[2] * image.unit_size[1]


def fwhm_z(image):
    popt, pcov = _gaussian_fit_z(image.data)
    return 2.35 * popt[2] * image.unit_size[2]


def fwtm_z(image):
    popt, pcov = _gaussian_fit_z(image.data)
    return 4.29 * popt[2] * image.unit_size[2]


def sd_rc(mask, image):
    center = np.where(image.data == np.max(image.data))
    center = [c[0] for c in center]
    print(center)
    num_pixel = 10 // image.unit_size[2]
    lineprofile = image.data[center[0], center[1],
                  center[2] - int(num_pixel / 2):center[2] - int(num_pixel / 2) + num_pixel]
    return np.max(image.data) / _mean_value(mask, image.data, 2) * np.sqrt(
        (np.std(lineprofile, ddof = 1) / np.mean(lineprofile)) ** 2 + (
                _standard_deviation(mask, image.data, 2) / _mean_value(mask, image.data, 2)) ** 2)
