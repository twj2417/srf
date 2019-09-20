# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: pet_cylindrical_scanner.py
@date: 7/1/2019
@desc:
'''
from numpy import pi


class PropertyClass:
    @property
    def average_radius(self):
        return (self.inner_radius + self.outer_radius) / 2

    @property
    def angle_per_rsector(self):
        return 2 * pi / self.nb_rsector

    @property
    def gap_crystal(self):
        return [mv - sz for mv, sz in zip(self.mv_crystal, self.sz_crystal)]

    @property
    def sz_submodule(self):
        sy = self.sz_crystal[0] + self.mv_crystal[0] * (self.nb_crystal[0] - 1)
        sz = self.sz_crystal[1] + self.mv_crystal[1] * (self.nb_crystal[1] - 1)
        return [sy, sz]

    @property
    def gap_submodule(self):
        return [mv - sz for mv, sz in zip(self.mv_submodule, self.sz_submodule)]

    @property
    def sz_module(self):
        sy = self.sz_submodule[0] + self.mv_submodule[0] * (self.nb_submodule[0] - 1)
        sz = self.sz_submodule[1] + self.mv_submodule[1] * (self.nb_submodule[1] - 1)
        return [sy, sz]

    @property
    def gap_module(self):
        return [mv - sz for mv, sz in zip(self.mv_module, self.sz_module)]

    @property
    def sz_rsector(self):
        sy = self.sz_module[0] + self.mv_module[0] * (self.nb_module[0] - 1)
        sz = self.sz_module[1] + self.mv_module[1] * (self.nb_module[1] - 1)
        return [sy, sz]

    @property
    def axial_length(self):
        return self.sz_rsector[1]

    @property
    def nb_crystal_per_submodule(self):
        return self.nb_crystal[0] * self.nb_crystal[1]

    @property
    def nb_submodule_per_module(self):
        return self.nb_submodule[0] * self.nb_submodule[1]

    @property
    def nb_module_per_rsector(self):
        return self.nb_module[0] * self.nb_module[1]

    @property
    def nb_crystal_per_module(self):
        return self.nb_crystal_per_submodule * self.nb_submodule_per_module

    @property
    def nb_crystal_per_rsector(self):
        return self.nb_crystal_per_submodule * self.nb_submodule_per_module * self.nb_module_per_rsector

    @property
    def nb_crystals(self):
        return self.nb_crystal_per_rsector * self.nb_rsector
