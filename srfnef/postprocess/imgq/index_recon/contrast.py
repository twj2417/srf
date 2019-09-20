import numpy as np

"""
Here we have contrast,CNR,CRC,SE,NSD,AI
"""


def contrast_hot(mask, image, ratio_in_phantom):
    return (_mean_value(mask, image, 1) / _mean_value(mask, image, 2) - 1) / (ratio_in_phantom - 1)


def contrast_cold(mask, image):
    return 1 - _mean_value(mask, image, 1) / _mean_value(mask, image, 2)


def crc1(mask, image):
    index_roi = np.where(mask == 1)
    cm = np.max(image[index_roi[0], index_roi[1]])
    return cm


def crc2(mask, image, ratio_in_phantom):
    return (_mean_value(mask, image, 1) / _mean_value(mask, image, 2)) / ratio_in_phantom


def cnr1(mask, image):
    return (_mean_value(mask, image, 1) - _mean_value(mask, image, 2)) / _standard_deviation(mask,
                                                                                             image,
                                                                                             2)


def cnr2(mask, image):
    return _mean_value(mask, image, 1) / _standard_deviation(mask, image, 2)


def _standard_deviation(mask, image, value):
    index_roi = np.where(mask == value)
    return np.std(image[index_roi[0], index_roi[1]], ddof = 1)


def _mean_value(mask, image, value):
    index_roi = np.where(mask == value)
    return np.mean(image[index_roi[0], index_roi[1]])


def snr1(mask, image):
    return _mean_value(mask, image, 1) / _standard_deviation(mask, image, 2)


def se(mask, image, num_bg):
    return _standard_deviation(mask, image, 2) / np.sqrt(num_bg)


def nsd(mask, image):
    index_roi = np.where(mask == 1)
    return np.std(image[index_roi[0], index_roi[1]]) / np.mean(image[index_roi[0], index_roi[1]])


def ai(mask, image):
    return _mean_value(mask, image, 1) / _mean_value(mask, image, 2) - 1
