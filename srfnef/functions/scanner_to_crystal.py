# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: scanner_to_crystal.py
@date: 6/14/2019
@desc:
'''
from srfnef import nef_class
from srfnef.geometry import PetEcatScanner
import numpy as np


@nef_class
class ScannerToCrystal:
    def __call__(self, scanner: PetEcatScanner) -> np.ndarray:
        nb_crystals_per_ring = scanner.nb_blocks_per_ring * scanner.blocks.shape[1] * \
                               scanner.blocks.shape[2]
        nb_crystals = nb_crystals_per_ring * scanner.nb_rings
        lors = np.zeros((nb_crystals, 3), dtype = np.float32)

        x = np.ones(scanner.blocks.shape[1], ) * scanner.average_radius
        y = (np.arange(scanner.blocks.shape[1]) + 0.5) * scanner.blocks.unit_size[1] - \
            scanner.blocks.size[1] / 2
        z = (np.arange(scanner.blocks.shape[2]) + 0.5) * scanner.blocks.unit_size[2] - \
            scanner.blocks.size[2] / 2
        x1 = np.kron(x, [[1]] * scanner.blocks.shape[2]).ravel()
        y1 = np.kron(y, [[1]] * scanner.blocks.shape[2]).ravel()
        xx = np.kron(x1, [[1]] * scanner.nb_blocks_per_ring).ravel()
        yy = np.kron(y1, [[1]] * scanner.nb_blocks_per_ring).ravel()
        zz = np.kron(z, [1] * scanner.blocks.shape[1])
        theta = 2 * np.pi / scanner.nb_blocks_per_ring * np.arange(scanner.nb_blocks_per_ring)

        theta1 = np.kron(theta, [1] * scanner.blocks.shape[1] * scanner.blocks.shape[2])
        xx1 = xx * np.cos(theta1) - yy * np.sin(theta1)
        yy1 = xx * np.sin(theta1) + yy * np.cos(theta1)

        lors[:, 0] = np.kron(xx1, [[1]] * scanner.nb_rings).ravel()
        lors[:, 1] = np.kron(yy1, [[1]] * scanner.nb_rings).ravel()
        for i in range(scanner.nb_rings):
            lors[nb_crystals_per_ring * i:nb_crystals_per_ring * (i + 1), 2] = \
                np.kron(zz, [[1]] * scanner.nb_blocks_per_ring).ravel() - scanner.axial_length / 2 \
                + i * (scanner.blocks.size[2] + scanner.gap) + 0.5 * scanner.blocks.size[2]

        return lors
