# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: emap_generator_mixin.py
@date: 4/23/2019
@desc:
'''
import numpy as np
import srfnef as nef
from srfnef import nef_class
from srfnef.utils import tqdm
from srfnef.geometry import PetScanner, PetCylindricalScanner, PetEcatScanner
from srfnef.data import Image, Emap
from srfnef.functions import BackProject, ScannerToLors, LorsToListmode
from srfnef.ops.deform_mixins import DeformMixin
import tensorflow as tf

from srfnef.utils import declare_eager_execution

mem_limit = 1e7


@nef_class
class EcatEmapGenerator(DeformMixin):
    mode: str
    scanner: PetEcatScanner

    def __call__(self, image: Image):
        from srfnef import EcatIndexToCrystalPos
        if self.mode == 'full':
            declare_eager_execution()
            ind2pos = EcatIndexToCrystalPos(self.scanner)
            ind = np.arange(self.scanner.nb_crystals)
            pos1 = pos2 = ind2pos(ind)
            pos1_ = np.kron(pos1, [1] * pos2.size)
            pos2_ = np.kron(pos2, [[1]] * pos1.size).reshape(-1, 3)
            lors_data = np.hstack((pos1_, pos2_))
            listmode = LorsToListmode()(nef.Lors(lors_data))
            return Emap(**BackProject(mode = 'tf-eager')(listmode, image).asdict())
        elif self.mode == 'block':
            declare_eager_execution()
            single_block_scanner = self.scanner.update(nb_blocks_per_ring = 1)
            ind2pos = EcatIndexToCrystalPos(single_block_scanner)
            ind = np.arange(self.scanner.nb_crystals_per_block * self.scanner.nb_rings)
            pos1 = pos2 = ind2pos(ind)
            pos1_x = np.kron(pos1[:, 0], [1] * ind.size)
            pos1_y = np.kron(pos1[:, 1], [1] * ind.size)
            pos1_z = np.kron(pos1[:, 2], [1] * ind.size)
            pos1_ = np.vstack((pos1_x, pos1_y, pos1_z)).transpose()

            emap_data = np.zeros(image.shape, np.float32)
            emap_tf = Emap(data = tf.Variable(emap_data), center = image.center,
                           size = image.size)
            for d in tqdm(range(self.scanner.nb_blocks_per_ring)):
                angle = d * self.scanner.angle_per_block
                print(angle)
                pos2_x = np.kron(pos2[:, 0], [[1]] * ind.size).ravel()
                pos2_y = np.kron(pos2[:, 1], [[1]] * ind.size).ravel()
                pos2_z = np.kron(pos2[:, 2], [[1]] * ind.size).ravel()
                pos2_ = np.vstack((pos2_x * np.cos(angle) - pos2_y * np.sin(angle),
                                   pos2_x * np.sin(angle) + pos2_y * np.cos(angle),
                                   pos2_z)).transpose()
                lors_data = np.hstack((pos1_, pos2_)).astype(np.float32)
                listmode = LorsToListmode()(nef.Lors(lors_data))
                listmode_tf = listmode.update(data = tf.Variable(listmode.data),
                                              lors = nef.Lors(tf.Variable(lors_data)))
                _emap = BackProject(mode = 'tf')(listmode_tf, emap_tf)
                for i in range(self.scanner.nb_blocks_per_ring):
                    _emap_rotate_data = self._rotate_tf(_emap.data,
                                                        i * self.scanner.angle_per_block)
                    tf.compat.v1.assign_add(emap_tf.data, _emap_rotate_data)
            emap_data = emap_tf.data.numpy()
            return emap_tf.update(data = emap_data, center = image.center, size = image.size)

        elif self.mode == 'block-full':
            declare_eager_execution()
            single_block_scanner = self.scanner.update(nb_blocks_per_ring = 1)
            ind2pos = EcatIndexToCrystalPos(single_block_scanner)
            ind = np.arange(self.scanner.nb_crystals_per_block * self.scanner.nb_rings)
            pos1 = pos2 = ind2pos(ind)

            emap_data = np.zeros(image.shape, np.float32)
            emap_tf = Emap(data = tf.Variable(emap_data), center = image.center,
                           size = image.size)
            for i in tqdm(range(self.scanner.nb_blocks_per_ring)):
                angle1 = i * self.scanner.angle_per_block
                pos1_x = np.kron(pos1[:, 0], [1] * ind.size)
                pos1_y = np.kron(pos1[:, 1], [1] * ind.size)
                pos1_z = np.kron(pos1[:, 2], [1] * ind.size)
                pos1_ = np.vstack((pos1_x * np.cos(angle1) - pos1_y * np.sin(angle1),
                                   pos1_x * np.sin(angle1) + pos1_y * np.cos(angle1),
                                   pos1_z)).transpose()
                for j in range(self.scanner.nb_blocks_per_ring):
                    angle2 = j * self.scanner.angle_per_block
                    pos2_x = np.kron(pos2[:, 0], [[1]] * ind.size).ravel()
                    pos2_y = np.kron(pos2[:, 1], [[1]] * ind.size).ravel()
                    pos2_z = np.kron(pos2[:, 2], [[1]] * ind.size).ravel()
                    pos2_ = np.vstack((pos2_x * np.cos(angle2) - pos2_y * np.sin(angle2),
                                       pos2_x * np.sin(angle2) + pos2_y * np.cos(angle2),
                                       pos2_z)).transpose()

                    lors_data = np.hstack((pos1_, pos2_)).astype(np.float32)
                    listmode = LorsToListmode()(nef.Lors(lors_data))
                    listmode_tf = listmode.update(data = tf.Variable(listmode.data),
                                                  lors = nef.Lors(tf.Variable(lors_data)))
                    _emap = BackProject(mode = 'tf')(listmode_tf, emap_tf)
                    tf.compat.v1.assign_add(emap_tf.data, _emap.data)
            emap_data = emap_tf.data.numpy()
            return emap_tf.update(data = emap_data, center = image.center, size = image.size)
        elif self.mode == 'rsector':
            return self.update(mode = 'block')(image)
        elif self.mode == 'rsector-full':
            return self.update(mode = 'block-full')(image)
        else:
            raise NotImplementedError


@nef_class
class CylindricalEmapGenerator(DeformMixin):
    mode: str
    scanner: PetCylindricalScanner

    def __call__(self, image: Image):
        from srfnef import CylindricalIndexToCrystalPos
        if self.mode == 'full':
            declare_eager_execution()
            ind2pos = CylindricalIndexToCrystalPos(self.scanner)
            ind = np.arange(self.scanner.nb_crystals)
            pos1 = pos2 = ind2pos(ind)
            pos1_ = np.kron(pos1, [1] * pos2.size)
            pos2_ = np.kron(pos2, [[1]] * pos1.size).reshape(-1, 3)
            lors_data = np.hstack((pos1_, pos2_))
            listmode = LorsToListmode()(nef.Lors(lors_data))
            return Emap(**BackProject(mode = 'tf-eager')(listmode, image).asdict())
        elif self.mode == 'rsector':
            declare_eager_execution()
            single_block_scanner = self.scanner.update(nb_rsector = 1)
            ind2pos = CylindricalIndexToCrystalPos(single_block_scanner)
            ind = np.arange(self.scanner.nb_crystal_per_rsector)
            pos1 = pos2 = ind2pos(ind)
            pos1_x = np.kron(pos1[:, 0], [1] * ind.size)
            pos1_y = np.kron(pos1[:, 1], [1] * ind.size)
            pos1_z = np.kron(pos1[:, 2], [1] * ind.size)
            pos1_ = np.vstack((pos1_x, pos1_y, pos1_z)).transpose()

            emap_data = np.zeros(image.shape, np.float32)
            emap_tf = Emap(data = tf.Variable(emap_data), center = image.center,
                           size = image.size)
            for d in tqdm(range(self.scanner.nb_rsector)):
                angle = d * self.scanner.angle_per_rsector
                pos2_x = np.kron(pos2[:, 0], [[1]] * ind.size).ravel()
                pos2_y = np.kron(pos2[:, 1], [[1]] * ind.size).ravel()
                pos2_z = np.kron(pos2[:, 2], [[1]] * ind.size).ravel()
                pos2_ = np.vstack((pos2_x * np.cos(angle) - pos2_y * np.sin(angle),
                                   pos2_x * np.sin(angle) + pos2_y * np.cos(angle),
                                   pos2_z)).transpose()
                lors_data = np.hstack((pos1_, pos2_)).astype(np.float32)
                listmode = LorsToListmode()(nef.Lors(lors_data))
                _emap = BackProject(mode = 'tf')(listmode, emap_tf)
                for i in range(self.scanner.nb_rsector):
                    _emap_rotate_data = self._rotate_tf(_emap.data,
                                                        i * self.scanner.angle_per_rsector)
                    tf.compat.v1.assign_add(emap_tf.data, _emap_rotate_data)
            emap_data = emap_tf.data.numpy()
            return emap_tf.update(data = emap_data, center = image.center, size = image.size)

        elif self.mode == 'rsector-full':
            declare_eager_execution()
            single_block_scanner = self.scanner.update(nb_rsector = 1)
            ind2pos = CylindricalIndexToCrystalPos(single_block_scanner)
            ind = np.arange(self.scanner.nb_crystal_per_rsector)
            pos1 = pos2 = ind2pos(ind)

            emap_data = np.zeros(image.shape, np.float32)
            emap_tf = Emap(data = tf.Variable(emap_data), center = image.center,
                           size = image.size)
            for i in tqdm(range(self.scanner.nb_rsector)):
                angle1 = i * self.scanner.angle_per_rsector
                pos1_x = np.kron(pos1[:, 0], [1] * ind.size)
                pos1_y = np.kron(pos1[:, 1], [1] * ind.size)
                pos1_z = np.kron(pos1[:, 2], [1] * ind.size)
                pos1_ = np.vstack((pos1_x * np.cos(angle1) - pos1_y * np.sin(angle1),
                                   pos1_x * np.sin(angle1) + pos1_y * np.cos(angle1),
                                   pos1_z)).transpose().astype(np.float32)
                for j in range(self.scanner.nb_rsector):
                    angle2 = j * self.scanner.angle_per_rsector
                    pos2_x = np.kron(pos2[:, 0], [[1]] * ind.size).ravel()
                    pos2_y = np.kron(pos2[:, 1], [[1]] * ind.size).ravel()
                    pos2_z = np.kron(pos2[:, 2], [[1]] * ind.size).ravel()
                    pos2_ = np.vstack((pos2_x * np.cos(angle2) - pos2_y * np.sin(angle2),
                                       pos2_x * np.sin(angle2) + pos2_y * np.cos(angle2),
                                       pos2_z)).transpose()

                    lors_data = np.hstack((pos1_, pos2_)).astype(np.float32)
                    listmode = LorsToListmode()(nef.Lors(lors_data))
                    _emap = BackProject(mode = 'tf')(listmode, emap_tf)
                    tf.compat.v1.assign_add(emap_tf.data, _emap.data)
            emap_data = emap_tf.data.numpy()
            return emap_tf.update(data = emap_data, center = image.center, size = image.size)
        elif self.mode == 'auto':
            declare_eager_execution()
            single_block_scanner = self.scanner.update(nb_rsector = 1)
            ind2pos = CylindricalIndexToCrystalPos(single_block_scanner)
            num_rsector = int(np.sqrt(mem_limit // (self.scanner.nb_crystal_per_rsector ** 2)))
            while not self.scanner.nb_rsector % num_rsector == 0:
                num_rsector -= 1
            ind = np.arange(self.scanner.nb_crystal_per_rsector * num_rsector)
            pos1 = pos2 = ind2pos(ind)

            emap_data = np.zeros(image.shape, np.float32)
            emap_tf = Emap(data = tf.Variable(emap_data), center = image.center,
                           size = image.size)
            for i in tqdm(range(0, self.scanner.nb_rsector, num_rsector)):
                angle1 = i * self.scanner.angle_per_rsector
                pos1_x = np.kron(pos1[:, 0], [1] * ind.size)
                pos1_y = np.kron(pos1[:, 1], [1] * ind.size)
                pos1_z = np.kron(pos1[:, 2], [1] * ind.size)
                pos1_ = np.vstack((pos1_x * np.cos(angle1) - pos1_y * np.sin(angle1),
                                   pos1_x * np.sin(angle1) + pos1_y * np.cos(angle1),
                                   pos1_z)).transpose().astype(np.float32)
                for j in range(0, self.scanner.nb_rsector, num_rsector):
                    angle2 = j * self.scanner.angle_per_rsector
                    pos2_x = np.kron(pos2[:, 0], [[1]] * ind.size).ravel()
                    pos2_y = np.kron(pos2[:, 1], [[1]] * ind.size).ravel()
                    pos2_z = np.kron(pos2[:, 2], [[1]] * ind.size).ravel()
                    pos2_ = np.vstack((pos2_x * np.cos(angle2) - pos2_y * np.sin(angle2),
                                       pos2_x * np.sin(angle2) + pos2_y * np.cos(angle2),
                                       pos2_z)).transpose()

                    lors_data = np.hstack((pos1_, pos2_)).astype(np.float32)
                    listmode = LorsToListmode()(nef.Lors(lors_data))
                    _emap = BackProject(mode = 'tf')(listmode, emap_tf)
                    tf.compat.v1.assign_add(emap_tf.data, _emap.data)
            emap_data = emap_tf.data.numpy()
            return emap_tf.update(data = emap_data, center = image.center, size = image.size)
        elif self.mode == 'block':
            return self.update(mode = 'rsector')(image)
        elif self.mode == 'block-full':
            return self.update(mode = 'rsector-full')(image)
        else:
            raise NotImplementedError


@nef_class
class EmapGenerator(DeformMixin):
    mode: str
    scanner: PetEcatScanner

    def __call__(self, *args, **kwargs):
        if isinstance(self.scanner, PetEcatScanner):
            return EcatEmapGenerator(self.mode, self.scanner)(*args, **kwargs)
        elif isinstance(self.scanner, PetCylindricalScanner):
            return CylindricalEmapGenerator(self.mode, self.scanner)(*args, **kwargs)
        else:
            raise NotImplementedError
