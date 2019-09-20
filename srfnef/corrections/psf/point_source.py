# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: point_source.py
@date: 5/7/2019
@desc:
'''

from srfnef import nef_class, List, Any
from srfnef.ops.common.property_mixins import ShapePropertyMixin, UnitSizePropertyMixin, \
    CentralSlicesPropertyMixin, CentralProfilesPropertyMixin
from srfnef.ops.common.imshow_mixin import ImshowMixin
from srfnef.ops.common.arithmetic_mixins import ArithmeticMixin
from srfnef.ops.common.magic_method_mixins import GetItemMixin
from srfnef.io import LoadMixin, SaveMixin
import attr


@nef_class
class PointSource(ShapePropertyMixin,
                  UnitSizePropertyMixin,
                  GetItemMixin,
                  CentralSlicesPropertyMixin,
                  CentralProfilesPropertyMixin,
                  LoadMixin,
                  SaveMixin,
                  ImshowMixin,
                  ArithmeticMixin):
    data: Any
    center: List(float, 3)
    size: List(float, 3)
    pos: List(int)
    psf_type: str = attr.ib(default = 'xy')
