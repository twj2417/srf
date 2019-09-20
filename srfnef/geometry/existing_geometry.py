# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: existing_geometry.py
@date: 5/21/2019
@desc:
'''

from .block import Block
from .pet_ecat_scanner import PetEcatScanner
from .image_config import ImageConfig

block = Block([20.0, 51.3, 51.3], [1, 15, 15])
scanner_1_4m = PetEcatScanner(400.0, 420.0, 26, 48, 3.42, block)

block = Block([20.0, 53.3, 53.3], [1, 13, 13])
scanner_mct = PetEcatScanner(424.5, 444.5, 4, 48, 0.0, block)

shape = [200, 200, 200]
center = [0., 0., 0.]
size = [410.0, 410.0, 410.0]

image_config_mct = ImageConfig(shape, center, size)

from .pet_cylindrical_scanner import PetCylindricalScanner

scanner_8panel = PetCylindricalScanner(146.7, 166.7, 8, [2, 2], [60, 80], [3, 4], [20, 20],
                                       [6, 6], [3.2, 3.2], [3, 3])

scanner_20panel = PetCylindricalScanner(389.72, 411.9, 20, [2, 2], [60, 80], [3, 4], [20, 20],
                                        [6, 6], [3.2, 3.2], [3, 3])
