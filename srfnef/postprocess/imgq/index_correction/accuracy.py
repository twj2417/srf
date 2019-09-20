from ..index_recon.contrast import _mean_value

"""
Here we have scatter/attenuation correction accuracy
"""


def accuracy(mask, image):
    return _mean_value(mask, image, 1) / _mean_value(mask, image, 2) * 100
