# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: scanner_to_lors.py
@date: 4/26/2019
@desc:
'''
from srfnef import nef_class
from srfnef.geometry import PetEcatScanner
from srfnef.data import Lors
from srfnef.ops.lors_generate_mixin import LorsGenerateMixin


@nef_class
class ScannerToLors(LorsGenerateMixin):
    mode: str

    def __call__(self, scanner: PetEcatScanner, d: int = 0) -> Lors:
        if self.mode == 'full':
            lors_data = self._mesh_crystal_full(scanner.nb_rings,
                                                scanner.blocks.shape,
                                                scanner.average_radius,
                                                scanner.blocks.size,
                                                scanner.nb_blocks_per_ring)
            lors_data[:, :3] += scanner.center
            lors_data[:, 3:6] += scanner.center
            return Lors(lors_data)
        elif self.mode == 'thin-ring':
            lors_data = self._mesh_crystal_thin_ring(scanner.blocks.shape,
                                                     scanner.average_radius,
                                                     scanner.blocks.size,
                                                     scanner.nb_blocks_per_ring,
                                                     scanner.gap,
                                                     d)
            lors_data[:, :3] += scanner.center
            lors_data[:, 3:6] += scanner.center
            return Lors(lors_data)
        elif self.mode == 'ring':
            lors_data = self._mesh_crystal_ring2(scanner.blocks.shape,
                                                 scanner.average_radius,
                                                 scanner.blocks.size,
                                                 scanner.nb_blocks_per_ring)
            lors_data[:, :3] += scanner.center
            lors_data[:, 3:6] += scanner.center
            return Lors(lors_data)
        elif self.mode == 'block-full':
            lors_data = self._mesh_crystal_ring_full(scanner.blocks.shape,
                                                     scanner.average_radius,
                                                     scanner.blocks.size,
                                                     scanner.nb_blocks_per_ring,
                                                     scanner.axial_length)
            lors_data[:, :3] += scanner.center
            lors_data[:, 3:6] += scanner.center
            return Lors(lors_data)
        else:
            raise NotImplementedError
