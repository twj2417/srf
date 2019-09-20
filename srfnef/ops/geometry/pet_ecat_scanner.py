# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: pet_ecat_scanner.py
@date: 6/28/2019
@desc:
'''
from numpy import pi


class PropertyClass:
    @property
    def average_radius(self):
        return (self.inner_radius + self.outer_radius) / 2

    @property
    def axial_length(self):
        return self.blocks.size[2] * self.nb_rings + self.gap * (self.nb_rings - 1)

    @property
    def nb_crystals_per_block(self):
        return self.blocks.shape[1] * self.blocks.shape[2]

    @property
    def nb_crystals_per_ring(self):
        return self.nb_crystals_per_block * self.nb_blocks_per_ring

    @property
    def nb_crystals(self):
        return self.nb_crystals_per_ring * self.nb_rings

    @property
    def angle_per_block(self):
        return 2 * pi / self.nb_blocks_per_ring

    # thin_ring functions should be abandoned
    @property
    def nb_thin_rings(self):
        return self.nb_rings * self.blocks.shape[2]

    @property
    def nb_crystals_per_thin_ring(self):
        return self.blocks.shape[1] * self.nb_blocks_per_ring
