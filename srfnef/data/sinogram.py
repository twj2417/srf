# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: srf_ct
@file: sinogram.py
@date: 3/20/2019
@desc:
'''

from srfnef import nef_class, Any
from srfnef.geometry.pet_ecat_scanner import PetEcatScanner
from srfnef.ops.common.property_mixins import ShapePropertyMixin


@nef_class
class Sinogram(ShapePropertyMixin):
    data: Any
    scanner: PetEcatScanner
