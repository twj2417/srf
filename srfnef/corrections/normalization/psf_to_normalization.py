# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: psf_to_normalization.py
@date: 5/8/2019
@desc:
'''

from srfnef import nef_class
from srfnef.corrections import FittedX, FittedZ
from .amp_x import AmplitudeX
from .amp_z import AmplitudeZ


@nef_class
class PsfToNormalizationX:
    def __call__(self, fitted_x: FittedX) -> AmplitudeX:
        return AmplitudeX(fitted_x.data[:, [0, 2]])


@nef_class
class PsfToNormalizationZ:
    def __call__(self, fitted_z: FittedZ) -> AmplitudeZ:
        return AmplitudeZ(fitted_z.data[:, [0, 2]])
