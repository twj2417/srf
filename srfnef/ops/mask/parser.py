# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: parser.py
@date: 6/26/2019
@desc:
'''
from srfnef import nef_class
import numpy as np

_eps = 1e-3


@nef_class
class MaskParser:
    '''
    mask image parser, output boolean mask image
    the attr mask consist several parts
    0 is for background
    negative values are for background in phantom(denote as phantom), where
    -1 is for the background, in which
    -2 to -Inf are for regions in background

    positive values are for ROIs, where
    integers are for hot-sources and
    int + 0.5 are for cold-sources
    '''

    def _get_hot_source(self, mask: np.ndarray):
        return np.logical_and(mask > 0, np.abs(mask - np.floor(mask)) <
                              _eps)

    def _get_cold_source(self, mask: np.ndarray):
        return np.logical_and(mask > 0,
                              np.abs(mask - 0.5 - np.floor(mask)) < _eps)

    def _get_single_hot_source(self, mask: np.ndarray):
        new_mask = mask * self._get_hot_source(mask)
        inds = np.unique(new_mask)
        out_inds = []
        out_mask = []
        for ind in inds[:]:
            if ind == 0:
                continue
            out_inds += [ind]
            out_mask += [new_mask == ind]
        return zip(out_inds, out_mask)

    def _get_single_cold_source(self, mask: np.ndarray):
        new_mask = mask * self._get_cold_source(mask)
        inds = np.unique(new_mask)
        out_inds = []
        out_mask = []
        for ind in inds[:]:
            if ind == 0:
                continue
            out_inds += [ind]
            out_mask += [new_mask == ind]
        return zip(out_inds, out_mask)

    def _get_background(self, mask: np.ndarray):
        return mask == 0

    def _get_phantom(self, mask: np.ndarray):
        '''get phantom but ROI'''
        return mask < 0

    def _get_single_phantom(self, mask: np.ndarray):
        new_mask = mask * self._get_phantom(mask)
        inds = np.unique(new_mask)
        out_inds = []
        out_mask = []
        for ind in inds[:]:
            if ind >= -1:
                continue
            out_inds += [ind]
            out_mask += [new_mask == ind]
        return zip(out_inds, out_mask)

    def _get_phantom_except(self, mask: np.ndarray):
        return mask == -1
