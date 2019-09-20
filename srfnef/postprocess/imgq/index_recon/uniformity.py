import numpy as np

"""
Here we have uniformity
"""


def positive_deviation(mask, image):
    L = mask == 1.0
    mean = np.mean(image[L])
    return 100 * (np.max(image[L]) - mean) / mean


def negative_deviation(mask, image):
    L = mask == 1.0
    mean = np.mean(image[L])
    return 100 * (mean - np.min(image[L])) / mean
