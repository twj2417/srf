# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: listmode.py
@date: 4/26/2019
@desc:
'''
from .lors import Lors
from srfnef import nef_class
from srfnef.ops.common.property_mixins import ShapePropertyMixin, LengthPropertyMixin
from srfnef.ops.common.magic_method_mixins import GetItemMixin
from srfnef.ops.common.arithmetic_mixins import ArithmeticMixin


@nef_class
class Listmode(ShapePropertyMixin,
               LengthPropertyMixin,
               GetItemMixin,
               ArithmeticMixin):
    data: object
    lors: Lors

    def append(self, listmode):
        import numpy as np
        lors = self.lors.append(listmode.lors)
        return self.update(data = np.hstack((self.data, listmode.data)), lors = lors)
