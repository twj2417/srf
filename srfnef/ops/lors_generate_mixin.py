# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: lors_generate_mixin.py
@date: 5/21/2019
@desc:
'''

import numpy as np
from numba import jit
from srfnef import List


class LorsGenerateMixin:
    @jit
    def _mesh_crystal_full(self, nb_rings: int,
                           block_shape: List(int, 3),
                           average_radius: int,
                           block_size: List(float, 3),
                           nb_blocks_per_ring: int):
        if nb_rings > 1:
            raise ValueError('Please use _mesh_crystal_ring instead')
        block_unit_size = [s1 / s2 for s1, s2 in zip(block_size, block_shape)]
        nb_crystals_per_ring = nb_blocks_per_ring * block_shape[1] * block_shape[2]
        nb_crystals = nb_crystals_per_ring * nb_rings
        lors = np.zeros((nb_crystals * nb_crystals, 6), dtype = np.float32)

        x = np.ones(block_shape[1], ) * average_radius
        y = (np.arange(block_shape[1]) + 0.5) * block_unit_size[1] - \
            block_size[1] / 2
        z = (np.arange(block_shape[2]) + 0.5) * block_unit_size[2] - \
            block_size[2] / 2
        xx = np.kron(x, [1] * nb_blocks_per_ring)
        yy = np.kron(y, [1] * nb_blocks_per_ring)
        theta = 2 * np.pi / nb_blocks_per_ring * np.arange(nb_blocks_per_ring)
        theta1 = np.kron(theta, [[1]] * block_shape[1]).ravel()
        xx1 = xx * np.cos(theta1) - yy * np.sin(theta1)
        yy1 = xx * np.sin(theta1) + yy * np.cos(theta1)
        xd = np.kron(xx1, [[1]] * block_shape[2]).ravel()
        yd = np.kron(yy1, [[1]] * block_shape[2]).ravel()
        zd = np.kron(z, [1] * block_shape[1] * nb_blocks_per_ring)

        lors[:, 0] = np.kron(xd, [1] * nb_crystals_per_ring)
        lors[:, 1] = np.kron(yd, [1] * nb_crystals_per_ring)
        lors[:, 2] = np.kron(zd, [1] * nb_crystals_per_ring)
        lors[:, 3] = np.kron(xd, [[1]] * nb_crystals_per_ring).ravel()
        lors[:, 4] = np.kron(yd, [[1]] * nb_crystals_per_ring).ravel()
        lors[:, 5] = np.kron(zd, [[1]] * nb_crystals_per_ring).ravel()

        return lors

    @jit
    def _mesh_crystal_ring(self, block_shape: List(int, 3),
                           average_radius: int,
                           block_size: List(float, 3),
                           nb_blocks_per_ring: int,
                           gap: float,
                           d: int):
        block_unit_size = [s1 / s2 for s1, s2 in zip(block_size, block_shape)]
        nb_crystals_per_ring = nb_blocks_per_ring * block_shape[1] * block_shape[2]

        lors = np.zeros((nb_crystals_per_ring * nb_crystals_per_ring, 6),
                        dtype = np.float32)

        x = np.ones(block_shape[1], ) * average_radius
        y = (np.arange(block_shape[1]) + 0.5) * block_unit_size[1] - \
            block_size[1] / 2
        z = (np.arange(block_shape[2]) + 0.5) * block_unit_size[2] - \
            block_size[2] * (d + 1) / 2
        xx = np.kron(x, [1] * nb_blocks_per_ring)
        yy = np.kron(y, [1] * nb_blocks_per_ring)
        theta = 2 * np.pi / nb_blocks_per_ring * np.arange(nb_blocks_per_ring)
        theta1 = np.kron(theta, [[1]] * block_shape[1]).ravel()
        xx1 = xx * np.cos(theta1) - yy * np.sin(theta1)
        yy1 = xx * np.sin(theta1) + yy * np.cos(theta1)
        xd = np.kron(xx1, [[1]] * block_shape[2]).ravel()
        yd = np.kron(yy1, [[1]] * block_shape[2]).ravel()
        zd = np.kron(z, [1] * block_shape[1] * nb_blocks_per_ring)

        lors[:, 0] = np.kron(xd, [1] * nb_crystals_per_ring)
        lors[:, 1] = np.kron(yd, [1] * nb_crystals_per_ring)
        lors[:, 2] = np.kron(zd, [1] * nb_crystals_per_ring)
        lors[:, 3] = np.kron(xd, [[1]] * nb_crystals_per_ring).ravel()
        lors[:, 4] = np.kron(yd, [[1]] * nb_crystals_per_ring).ravel()
        lors[:, 5] = np.kron(zd + d * (block_size[2] + gap),
                             [[1]] * nb_crystals_per_ring).ravel()

        return lors

    def _mesh_crystal_ring2(self, block_shape: List(int, 3),
                            average_radius: int,
                            block_size: List(float, 3),
                            nb_blocks_per_ring: int):
        nb_crystals_per_block = block_shape[1] * block_shape[2]
        nb_crystals_per_ring = nb_blocks_per_ring * nb_crystals_per_block
        nb_crystals_per_ring2 = (nb_blocks_per_ring - 1) * nb_crystals_per_block
        lors = np.zeros((nb_crystals_per_ring * nb_crystals_per_ring2 // 2, 6),
                        dtype = np.float32)

        block_unit_size = [s1 / s2 for s1, s2 in zip(block_size, block_shape)]

        x = np.ones(block_shape[1], ) * average_radius
        y = (np.arange(block_shape[1]) + 0.5) * block_unit_size[1] - \
            block_size[1] / 2
        z = (np.arange(block_shape[2]) + 0.5) * block_unit_size[2] - \
            block_size[2] / 2

        xx = np.kron(x, [[1]] * block_shape[2]).ravel()
        yy = np.kron(y, [[1]] * block_shape[2]).ravel()
        zz = np.kron(z, [1] * block_shape[1])

        theta_list = 2 * np.pi / nb_blocks_per_ring * np.arange(nb_blocks_per_ring)

        c = 0
        N = nb_crystals_per_block ** 2
        for theta1 in theta_list:
            xx1 = xx * np.cos(theta1) - yy * np.sin(theta1)
            yy1 = xx * np.sin(theta1) + yy * np.cos(theta1)
            zz1 = zz
            for theta2 in theta_list:
                if theta2 <= theta1:
                    continue
                xx2 = xx * np.cos(theta2) - yy * np.sin(theta2)
                yy2 = xx * np.sin(theta2) + yy * np.cos(theta2)
                zz2 = zz

                lors[c: c + N, 0] = np.kron(xx1, [1] * nb_crystals_per_block)
                lors[c: c + N, 1] = np.kron(yy1, [1] * nb_crystals_per_block)
                lors[c: c + N, 2] = np.kron(zz1, [1] * nb_crystals_per_block)
                lors[c: c + N, 3] = np.kron(xx2, [[1]] * nb_crystals_per_block).ravel()
                lors[c: c + N, 4] = np.kron(yy2, [[1]] * nb_crystals_per_block).ravel()
                lors[c: c + N, 5] = np.kron(zz2, [[1]] * nb_crystals_per_block).ravel()
                c += N

        return lors

    def _mesh_crystal_ring_full(self, block_shape: List(int, 3),
                                average_radius: int,
                                block_size: List(float, 3),
                                nb_blocks_per_ring: int,
                                axial_length: float):
        nb_crystals_per_block = block_shape[1] * block_shape[2]
        nb_crystals_per_ring = nb_blocks_per_ring * nb_crystals_per_block
        nb_crystals_per_ring2 = (nb_blocks_per_ring - 1) * nb_crystals_per_block
        lors = np.zeros((nb_crystals_per_ring * nb_crystals_per_ring2 // 2, 6),
                        dtype = np.float32)

        block_unit_size = [s1 / s2 for s1, s2 in zip(block_size, block_shape)]

        x = np.ones(block_shape[1], ) * average_radius
        y = (np.arange(block_shape[1]) + 0.5) * block_unit_size[1] - \
            block_size[1] / 2
        z = (np.arange(block_shape[2]) + 0.5) * block_unit_size[2]

        xx = np.kron(x, [[1]] * block_shape[2]).ravel()
        yy = np.kron(y, [[1]] * block_shape[2]).ravel()
        zz = np.kron(z, [1] * block_shape[1])

        theta_list = 2 * np.pi / nb_blocks_per_ring * np.arange(nb_blocks_per_ring)

        c = 0
        N = nb_crystals_per_block ** 2
        for theta1 in theta_list:
            xx1 = xx * np.cos(theta1) - yy * np.sin(theta1)
            yy1 = xx * np.sin(theta1) + yy * np.cos(theta1)
            zz1 = zz
            for theta2 in theta_list:
                if theta2 <= theta1:
                    continue
                xx2 = xx * np.cos(theta2) - yy * np.sin(theta2)
                yy2 = xx * np.sin(theta2) + yy * np.cos(theta2)
                zz2 = zz

                lors[c: c + N, 0] = np.kron(xx1, [1] * nb_crystals_per_block)
                lors[c: c + N, 1] = np.kron(yy1, [1] * nb_crystals_per_block)
                lors[c: c + N, 2] = np.kron(zz1, [1] * nb_crystals_per_block) - axial_length / 2
                lors[c: c + N, 3] = np.kron(xx2, [[1]] * nb_crystals_per_block).ravel()
                lors[c: c + N, 4] = np.kron(yy2, [[1]] * nb_crystals_per_block).ravel()
                lors[c: c + N, 5] = np.kron(zz2, [
                    [1]] * nb_crystals_per_block).ravel() - axial_length / 2
                c += N

        return lors

    @jit
    def _mesh_crystal_thin_ring(self, block_shape: List(int, 3),
                                average_radius: int,
                                block_size: List(float, 3),
                                nb_blocks_per_ring: int,
                                gap: float,
                                d: int):
        if gap > 0.0:
            raise ValueError('Please use _mesh_crystal_ring instead')
        nb_crystals_per_thin_ring = block_shape[1] * nb_blocks_per_ring
        block_unit_size = [s1 / s2 for s1, s2 in zip(block_size, block_shape)]

        lors = np.zeros((nb_crystals_per_thin_ring * nb_crystals_per_thin_ring, 6),
                        dtype = np.float32)

        x = np.ones(block_shape[1], ) * average_radius
        y = (np.arange(block_shape[1]) + 0.5) * block_unit_size[1] - \
            block_size[1] / 2
        z = 0.5 * block_unit_size[2] - block_unit_size[2] * (d + 1) / 2
        xx = np.kron(x, [1] * nb_blocks_per_ring)
        yy = np.kron(y, [1] * nb_blocks_per_ring)
        theta = 2 * np.pi / nb_blocks_per_ring * np.arange(nb_blocks_per_ring)
        theta1 = np.kron(theta, [[1]] * block_shape[1]).ravel()
        xd = xx * np.cos(theta1) - yy * np.sin(theta1)
        yd = xx * np.sin(theta1) + yy * np.cos(theta1)
        zd = np.kron(z, [1] * block_shape[1] * nb_blocks_per_ring)

        lors[:, 0] = np.kron(xd, [1] * nb_crystals_per_thin_ring)
        lors[:, 1] = np.kron(yd, [1] * nb_crystals_per_thin_ring)
        lors[:, 2] = np.kron(zd, [1] * nb_crystals_per_thin_ring)
        lors[:, 3] = np.kron(xd, [[1]] * nb_crystals_per_thin_ring).ravel()
        lors[:, 4] = np.kron(yd, [[1]] * nb_crystals_per_thin_ring).ravel()
        lors[:, 5] = np.kron(zd, [[1]] * nb_crystals_per_thin_ring).ravel() + d * \
                     block_unit_size[2]

        return lors
