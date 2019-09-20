# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: pet_cylindrical_scanner.py
@date: 6/28/2019
@desc:
'''
import numpy as np
from srfnef.geometry.pet_cylindrical_scanner import PetCylindricalScanner
from srfnef import nef_class, Listmode


@nef_class
class CylindricalCrystalPosToIndex:
    scanner: PetCylindricalScanner

    def __call__(self, pos: np.ndarray,
                 verbose: bool = False) -> (np.ndarray, tuple):
        irsector = self._position_to_rsector_index(pos, self.scanner)
        pos_ = self._rotate_to_rsector0(pos, self.scanner, irsector)

        imodule = self._position_to_module_index(pos_, self.scanner)

        pos_ = self._move_to_module0(pos_, self.scanner, imodule)

        isubmodule = self._position_to_submodule_index(pos_, self.scanner)

        pos_ = self._move_to_submodule0(pos_, self.scanner, isubmodule)
        icrystal = self._position_to_crystal_index(pos_, self.scanner)
        if verbose:
            return icrystal[0], icrystal[1], isubmodule[0], isubmodule[1], imodule[0], \
                   imodule[1], irsector
        else:
            return icrystal[0] + icrystal[1] * self.scanner.nb_crystal[0] \
                   + (isubmodule[0] + isubmodule[1] * self.scanner.nb_submodule[0]) * \
                   self.scanner.nb_crystal_per_submodule \
                   + (imodule[0] + imodule[1] * self.scanner.nb_module[
                1]) * self.scanner.nb_crystal_per_module \
                   + irsector * self.scanner.nb_crystal_per_rsector

    @staticmethod
    def _position_to_rsector_index(pos, scanner: PetCylindricalScanner):
        xc, yc = pos[:, 0], pos[:, 1]
        return np.round(np.arctan2(yc, xc) / scanner.angle_per_rsector).astype(
            int) % scanner.nb_rsector

    @staticmethod
    def _rotate_to_rsector0(pos, scanner: PetCylindricalScanner, irsector):
        angle = irsector * scanner.angle_per_rsector
        _pos = np.zeros(pos.shape)
        xc, yc = pos[:, 0], pos[:, 1]
        _pos[:, 0] = xc * np.cos(angle) + yc * np.sin(angle)
        _pos[:, 1] = -xc * np.sin(angle) + yc * np.cos(angle)
        _pos[:, 2] = pos[:, 2]
        return _pos

    @staticmethod
    def _position_to_module_index(pos, scanner: PetCylindricalScanner):
        yc = pos[:, 1]
        zc = pos[:, 2]
        indy = np.floor((yc + scanner.sz_rsector[0] / 2) / scanner.mv_module[0]).astype(int)
        indz = np.floor((zc + scanner.sz_rsector[1] / 2) / scanner.mv_module[1]).astype(int)
        return [indy, indz]

    @staticmethod
    def _move_to_module0(pos, scanner: PetCylindricalScanner, imodule):
        move_y = imodule[0] * scanner.mv_module[0] - scanner.sz_rsector[0] / 2 + \
                 scanner.sz_module[0] / 2
        move_z = imodule[1] * scanner.mv_module[1] - scanner.sz_rsector[1] / 2 + \
                 scanner.sz_module[1] / 2
        _pos = np.zeros(pos.shape)
        _pos[:, 0] = pos[:, 0]
        _pos[:, 1] = pos[:, 1] - move_y
        _pos[:, 2] = pos[:, 2] - move_z
        return _pos

    @staticmethod
    def _position_to_submodule_index(pos, scanner: PetCylindricalScanner):
        yc = pos[:, 1]
        zc = pos[:, 2]
        indy = np.floor((yc + scanner.sz_module[0] / 2) / scanner.mv_submodule[0]).astype(int)
        indz = np.floor((zc + scanner.sz_module[1] / 2) / scanner.mv_submodule[1]).astype(int)
        return [indy, indz]

    @staticmethod
    def _move_to_submodule0(pos, scanner: PetCylindricalScanner, isubmodule):
        move_y = isubmodule[0] * scanner.mv_submodule[0] - scanner.sz_module[0] / 2 + \
                 scanner.sz_submodule[0] / 2
        move_z = isubmodule[1] * scanner.mv_submodule[1] - scanner.sz_module[1] / 2 + \
                 scanner.sz_submodule[1] / 2
        _pos = np.zeros(pos.shape)
        _pos[:, 0] = pos[:, 0]
        _pos[:, 1] = pos[:, 1] - move_y
        _pos[:, 2] = pos[:, 2] - move_z
        return _pos

    @staticmethod
    def _position_to_crystal_index(pos, scanner: PetCylindricalScanner):
        yc = pos[:, 1]
        zc = pos[:, 2]
        indy = np.floor((yc + scanner.sz_submodule[0] / 2) / scanner.mv_crystal[0]).astype(int)
        indz = np.floor((zc + scanner.sz_submodule[1] / 2) / scanner.mv_crystal[1]).astype(int)
        return [indy, indz]


@nef_class
class CylindricalIndexToCrystalPos:
    scanner: PetCylindricalScanner

    def __call__(self, ind: (np.ndarray, tuple), verbose = False) -> np.ndarray:
        if verbose:
            icrystal_y, icrystal_z, isubmodule_y, isubmodule_z, imodule_y, imodule_z, irsector = ind
        else:
            icrystal_y = ind % self.scanner.nb_crystal[0]
            ind_ = ind // self.scanner.nb_crystal[0]
            icrystal_z = ind_ % self.scanner.nb_crystal[1]
            ind_ = ind_ // self.scanner.nb_crystal[1]
            isubmodule_y = ind_ % self.scanner.nb_submodule[0]
            ind_ = ind_ // self.scanner.nb_submodule[0]
            isubmodule_z = ind_ % self.scanner.nb_submodule[1]
            ind_ = ind_ // self.scanner.nb_submodule[1]
            imodule_y = ind_ % self.scanner.nb_module[0]
            ind_ = ind_ // self.scanner.nb_module[0]
            imodule_z = ind_ % self.scanner.nb_module[1]
            ind_ = ind_ // self.scanner.nb_module[1]
            irsector = ind_
        scanner = self.scanner

        pos = np.zeros((icrystal_y.size, 3), np.float32)
        x_crystal = scanner.average_radius
        y_crystal = icrystal_y * scanner.mv_crystal[0] + 0.5 * scanner.sz_crystal[0] \
                    - scanner.sz_submodule[0] / 2
        z_crystal = icrystal_z * scanner.mv_crystal[1] + 0.5 * scanner.sz_crystal[1] \
                    - scanner.sz_submodule[1] / 2

        y_submodule_mv = isubmodule_y * scanner.mv_submodule[0] - scanner.sz_module[0] / 2 + \
                         scanner.sz_submodule[0] / 2
        z_submodule_mv = isubmodule_z * scanner.mv_submodule[1] - scanner.sz_module[1] / 2 + \
                         scanner.sz_submodule[1] / 2

        y_submodule = y_crystal + y_submodule_mv
        z_submodule = z_crystal + z_submodule_mv

        y_module_mv = imodule_y * scanner.mv_module[0] - scanner.sz_rsector[0] / 2 + \
                      scanner.sz_module[0] / 2
        z_module_mv = imodule_z * scanner.mv_module[1] - scanner.sz_rsector[1] / 2 + \
                      scanner.sz_module[1] / 2

        y_module = y_submodule + y_module_mv
        z_module = z_submodule + z_module_mv

        x_module = x_crystal

        angle = scanner.angle_per_rsector * irsector

        pos[:, 0] = x_module * np.cos(angle) - y_module * np.sin(angle)
        pos[:, 1] = x_module * np.sin(angle) + y_module * np.cos(angle)
        pos[:, 2] = z_module
        return pos


@nef_class
class CylindricalLorsCompress:
    scanner: PetCylindricalScanner

    def __call__(self, listmode: Listmode):
        ind1 = CylindricalCrystalPosToIndex(self.scanner)(listmode.lors.data[:, :3])
        ind2 = CylindricalCrystalPosToIndex(self.scanner)(listmode.lors.data[:, 3:6])
        if listmode.lors.shape[1] == 6:
            tof = None
        else:
            tof = listmode.lors.data[:, 6]

        pos1 = CylindricalIndexToCrystalPos(self.scanner)(ind1)
        pos2 = CylindricalIndexToCrystalPos(self.scanner_8panel)(ind2)
        if tof is not None:
            lors_data = np.hstack((pos1, pos2, tof.reshape(-1, 1)))
        else:
            lors_data = np.hstack((pos1, pos2))
        return listmode.update(lors = lors_data)
