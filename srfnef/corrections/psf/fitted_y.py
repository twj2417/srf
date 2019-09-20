# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: fitted_x.py
@date: 5/7/2019
@desc:
'''
import numpy as np
from srfnef import nef_class
from srfnef.ops.common.property_mixins import ShapePropertyMixin, LengthPropertyMixin
from srfnef.ops.common.plot_mixin import PlotMixin


@nef_class
class FittedY(ShapePropertyMixin, LengthPropertyMixin, PlotMixin):
    data: np.ndarray

    @property
    def ay(self):
        return self.data[:, 0]

    @property
    def sigy(self):
        return self.data[:, 1]

    @property
    def uy(self):
        return self.data[:, 2]
