# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: image_config.py
@date: 4/26/2019
@desc:
'''
from srfnef import nef_class, List
from srfnef.ops.common.property_mixins import UnitSizePropertyMixin


@nef_class
class ImageConfig(UnitSizePropertyMixin):
    """
    Image data with center and size info.
    """

    shape: List(int, 3)
    center: List(float, 3)
    size: List(float, 3)
