# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: pet_ecat_scanner.py
@date: 6/28/2019
@desc:
'''
import numpy as np
from srfnef.geometry.pet_ecat_scanner import PetEcatScanner
from srfnef import nef_class


@nef_class
class EcatCrystalPosToIndex:
    scanner: PetEcatScanner

    def __call__(self, pos: np.ndarray,
                 verbose: bool = False) -> (np.ndarray, tuple):
        ''' cal'''
        if len(pos.shape) == 1:
            pos = pos.reshape((1, pos.size))

        iblock = self._position_to_block_index(pos, self.scanner)
        iring = self._position_to_ring_index(pos, self.scanner)
        iy = self._position_to_y_index_per_block(pos, self.scanner, iblock)
        iz = self._position_to_z_index_per_block(pos, self.scanner, iring)
        if verbose:
            return iy, iz, iblock, iring
        else:
            return iy + iz * self.scanner.blocks.shape[1] \
                   + iblock * self.scanner.nb_crystals_per_block \
                   + iring * self.scanner.nb_crystals_per_ring

    @staticmethod
    def _position_to_block_index(pos, scanner: PetEcatScanner):
        xc, yc = pos[:, 0], pos[:, 1]
        return np.round(np.arctan2(yc, xc) / scanner.angle_per_block).astype(
            int) % scanner.nb_blocks_per_ring

    @staticmethod
    def _position_to_ring_index(pos, scanner: PetEcatScanner):
        zc = pos[:, 2]
        return np.floor((zc + scanner.axial_length / 2) / (scanner.gap + scanner.blocks.size[
            2])).astype(int)

    @staticmethod
    def _position_to_y_index_per_block(pos, scanner: PetEcatScanner, iblock: int):
        angle = iblock * scanner.angle_per_block
        xc, yc = pos[:, 0], pos[:, 1]
        y_corr = -xc * np.sin(angle) + yc * np.cos(angle) + scanner.blocks.size[1] / 2

        return np.round(y_corr / scanner.blocks.unit_size[1] - 0.5).astype(int)

    @staticmethod
    def _position_to_z_index_per_block(pos, scanner: PetEcatScanner, iring: int):
        z_corr = pos[:, 2] + scanner.axial_length / 2 \
                 - iring * (scanner.blocks.size[2] + scanner.gap)
        return np.round(z_corr / scanner.blocks.unit_size[2] - 0.5).astype(int)


@nef_class
class EcatIndexToCrystalPos:
    scanner: PetEcatScanner

    def __call__(self, ind: (np.ndarray, tuple), verbose = False) -> np.ndarray:
        if verbose:
            iy, iz, iblock, iring = ind
        else:
            iy = ind % self.scanner.blocks.shape[1]
            iz = (ind // self.scanner.blocks.shape[1]) % self.scanner.blocks.shape[2]
            iblock = (ind // self.scanner.blocks.shape[1] // self.scanner.blocks.shape[
                2]) % self.scanner.nb_blocks_per_ring
            iring = ind // self.scanner.blocks.shape[1] // self.scanner.blocks.shape[
                2] // self.scanner.nb_blocks_per_ring

        pos = np.zeros((iy.size, 3), dtype = np.float32)
        x0 = self.scanner.average_radius
        y0 = (iy + 0.5) * self.scanner.blocks.unit_size[1] - self.scanner.blocks.size[1] / 2

        theta = self.scanner.angle_per_block * iblock

        pos[:, 0] = x0 * np.cos(theta) - y0 * np.sin(theta)
        pos[:, 1] = x0 * np.sin(theta) + y0 * np.cos(theta)
        pos[:, 2] = (iz + 0.5) * self.scanner.blocks.unit_size[2] \
                    + iring * (self.scanner.gap + self.scanner.blocks.size[2]) \
                    - self.scanner.axial_length / 2
        return pos
