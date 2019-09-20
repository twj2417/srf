# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: contrast_cold.py
@date: 6/26/2019
@desc:
'''
from srfnef import nef_class, Image
from srfnef.ops.mask.parser import MaskParser
import numpy as np


@nef_class
class ConstrastCold(MaskParser):
    def __call__(self, mask: np.ndarray, image: Image):
        if len(image.shape) > 2:
            image_ = image.central_slices[2]
        else:
            image_ = image.data
        mean_back = np.mean(image_[self._get_phantom(mask)])
        return [(ind, np.mean(image_[mk]) / mean_back) for ind, mk in
                self._get_single_cold_source(mask)]
