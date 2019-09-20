# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: pet_cylindrical.py
@date: 9/18/2019
@desc:
'''
from srfnef.functions.geometry.pet_cylindrical_scanner import CylindricalCrystalPosToIndex, \
    CylindricalIndexToCrystalPos
from srfnef import PetCylindricalScanner, nef_class, Listmode, Sinogram, Lors
import numpy as np
from scipy.sparse import coo_matrix


@nef_class
class PetCylindricalSinogramToListmode:
    scanner: PetCylindricalScanner

    def __call__(self, sino: Sinogram) -> Listmode:
        irow_, icol_ = sino.data.nonzero()
        data_ = sino.data.data
        pos1 = CylindricalIndexToCrystalPos(self.scanner)(irow_)
        pos2 = CylindricalIndexToCrystalPos(self.scanner)(icol_)
        lors_data = np.hstack((pos1, pos2))
        return Listmode(data_, Lors(lors_data))


@nef_class
class PetCylindricalListmodeToSinogram:
    scanner: PetCylindricalScanner

    def __call__(self, listmode: Listmode) -> Sinogram:
        pos1 = listmode.lors.data[:, :3]
        pos2 = listmode.lors.data[:, 3:6]
        irow_ = CylindricalCrystalPosToIndex(self.scanner)(pos1)
        icol_ = CylindricalCrystalPosToIndex(self.scanner)(pos2)
        N = self.scanner.nb_all_crystal
        sino_data = coo_matrix((listmode.data, (irow_, icol_)), shape = (N, N))
        return Sinogram(sino_data, self.scanner)


@nef_class
class PetCylindricalListmodeTrans:
    scanner: PetCylindricalScanner

    def __call__(self, listmode: Listmode) -> Listmode:
        pos1 = listmode.lors.data[:, :3]
        pos2 = listmode.lors.data[:, 3:6]
        ind1 = CylindricalCrystalPosToIndex(self.scanner)(pos1)
        ind2 = CylindricalCrystalPosToIndex(self.scanner)(pos2)
        pos1_new = CylindricalIndexToCrystalPos(self.scanner)(ind1)
        pos2_new = CylindricalIndexToCrystalPos(self.scanner)(ind2)
        if listmode.lors.data.shape[1] == 6:
            lors_data_new = np.hstack((pos1_new, pos2_new)).astype(np.float32)
        else:
            tof_val = listmode.lors.data[:, -1]
            lors_data_new = np.hstack((pos1_new, pos2_new, tof_val.reshape(-1, 1))).astype(
                np.float32)
        return listmode.update(lors = Lors(lors_data_new))
