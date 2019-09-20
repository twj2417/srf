# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: dvf.py
@date: 5/30/2019
@desc:
'''

from srfnef import nef_class, Any
from srfnef.io import LoadMixin, SaveMixin
from srfnef.ops.common.imshow_mixin import ImshowMixin
from srfnef.ops.common.property_mixins import ShapePropertyMixin


@nef_class
class DvfX(ImshowMixin, ShapePropertyMixin):
    data: Any


@nef_class
class DvfY(ImshowMixin, ShapePropertyMixin):
    data: Any


@nef_class
class DvfZ(ImshowMixin, ShapePropertyMixin):
    data: Any


@nef_class
class Dvf(LoadMixin,
          SaveMixin,
          ImshowMixin):
    """
    deformable vector fields
    """
    dvf_x: DvfX
    dvf_y: DvfY
    dvf_z: DvfZ
