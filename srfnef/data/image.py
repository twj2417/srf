# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: srf_ct
@file: image.py
@date: 3/20/2019
@desc:
'''

from srfnef import nef_class, List, Any
from srfnef.ops.common.property_mixins import ShapePropertyMixin, UnitSizePropertyMixin, \
    CentralSlicesPropertyMixin, CentralProfilesPropertyMixin, AverageProfilesPropertyMixin
from srfnef.ops.common.imshow_mixin import ImshowMixin
from srfnef.ops.common.arithmetic_mixins import ArithmeticMixin
from srfnef.ops.common.magic_method_mixins import GetItemMixin
from srfnef.io import LoadMixin, SaveMixin


@nef_class
class Image(ShapePropertyMixin,
            UnitSizePropertyMixin,
            GetItemMixin,
            CentralSlicesPropertyMixin,
            CentralProfilesPropertyMixin,
            AverageProfilesPropertyMixin,
            LoadMixin,
            SaveMixin,
            ImshowMixin,
            ArithmeticMixin):
    """
    Image data with center and size info.
    """

    data: Any
    center: List(float, 3)
    size: List(float, 3)
