# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: amp_z.py
@date: 5/8/2019
@desc:
'''

import numpy as np
from srfnef import nef_class
from srfnef.ops.common.property_mixins import ShapePropertyMixin, LengthPropertyMixin
from srfnef.mixins import PlotMixin


@nef_class
class AmplitudeZ(ShapePropertyMixin, LengthPropertyMixin, PlotMixin):
    data: np.ndarray

    @property
    def az(self):
        return self.data[:, 0]

    @property
    def uz(self):
        return self.data[:, 1]
