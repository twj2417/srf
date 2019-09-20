# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: block.py
@date: 4/26/2019
@desc:
'''
from srfnef import nef_class, List
from srfnef.ops.common.property_mixins import UnitSizePropertyMixin


@nef_class
class Block(UnitSizePropertyMixin):
    size: List(float, 3)
    shape: List(int, 3)
    interval: List(float, 3)

    def __attrs_post_init__(self):
        if self.interval is None:
            object.__setattr__(self, 'interval', [0, 0, 0])
