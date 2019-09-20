# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: crystal_to_id.py
@date: 6/14/2019
@desc:
'''

import numpy as np
from srfnef import nef_class
from srfnef.geometry import PetEcatScanner


@nef_class
class CrystalToId:
    def __call__(self, crystal_pos: np.ndarray, scanner: PetEcatScanner):
        iblock1 = _position_to_block_index(crystal_pos, scanner)
        iring1 = _position_to_ring_index(crystal_pos, scanner)
        _fst = _rotate_to_block0(crystal_pos, scanner, iblock1)
        iy1 = _position_to_y_index_per_block(_fst, scanner)
        iz1 = _position_to_z_index_per_block(_fst, scanner)
        return iy1 + iz1 * scanner.blocks.shape[
            1] + iblock1 * scanner.nb_crystals_per_block + scanner.nb_crystals_per_ring * iring1


def _position_to_block_index(pos, scanner: PetEcatScanner):
    xc, yc = pos[:, 0], pos[:, 1]
    return np.round(np.arctan2(yc, xc) / scanner.angle_per_block).astype(
        int) % scanner.nb_blocks_per_ring


def _position_to_thin_ring_index(pos, scanner: PetEcatScanner):
    zc = pos[:, 2]
    n_gap = (zc + scanner.axial_length /
             2) // (scanner.blocks.size[2] + scanner.gap)

    return np.floor(
        (zc + scanner.axial_length / 2 - n_gap * scanner.gap) / scanner.blocks.unit_size[
            2]).astype(int)


def _position_to_ring_index(pos, scanner: PetEcatScanner):
    zc = pos[:, 2]
    n_gap = (zc + scanner.axial_length /
             2) // (scanner.blocks.size[2] + scanner.gap)

    return np.floor(
        (zc + scanner.axial_length / 2 - n_gap * scanner.gap) / scanner.blocks.size[2]).astype(int)


def _rotate_to_block0(pos, scanner: PetEcatScanner, iblock):
    angle = iblock * scanner.angle_per_block
    _pos = np.zeros(pos.shape)
    xc, yc = pos[:, 0], pos[:, 1]
    _pos[:, 0] = xc * np.cos(angle) + yc * np.sin(angle)
    _pos[:, 1] = -xc * np.sin(angle) + yc * np.cos(angle)
    _pos[:, 2] = pos[:, 2]
    return _pos


def _position_to_y_index_per_block(pos, scanner: PetEcatScanner):
    return np.round(
        (pos[:, 1] + scanner.blocks.size[1] / 2) // scanner.blocks.unit_size[1]).astype(
        int)


def _position_to_z_index_per_block(pos, scanner: PetEcatScanner):
    z_corr = (pos[:, 2] + scanner.axial_length / 2) % scanner.blocks.size[2]
    return np.round(z_corr // scanner.blocks.unit_size[2]).astype(int)
