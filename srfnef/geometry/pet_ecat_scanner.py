# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: pet_ecat_scanner.py
@date: 4/26/2019
@desc:
'''
from srfnef import nef_class
from .block import Block
from .tof_config import TofConfig
from srfnef.geometry import PetScanner
from srfnef.ops.geometry.pet_ecat_scanner import PropertyClass


@nef_class
class PetEcatScanner(PetScanner, PropertyClass):
    inner_radius: float
    outer_radius: float
    nb_rings: int
    nb_blocks_per_ring: int
    gap: float
    blocks: Block
    tof: TofConfig
    # center: List(float, 3) = attr.ib(default = [0.0, 0.0, 0.0])
