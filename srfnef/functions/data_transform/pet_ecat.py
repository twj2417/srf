# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: pet_ecat.py
@date: 9/18/2019
@desc:
'''
from srfnef.functions.geometry.pet_ecat_scanner import EcatCrystalPosToIndex, EcatIndexToCrystalPos
from srfnef import PetEcatScanner, nef_class, Listmode, Sinogram, Lors
import numpy as np
from scipy.sparse import coo_matrix


@nef_class
class PetEcatSinogramToListmode:
    scanner: PetEcatScanner

    def __call__(self, sino: Sinogram) -> Listmode:
        irow_, icol_ = sino.data.nonzero()
        data_ = sino.data.data
        pos1 = EcatIndexToCrystalPos(self.scanner)(irow_)
        pos2 = EcatIndexToCrystalPos(self.scanner)(icol_)
        lors_data = np.hstack((pos1, pos2))
        return Listmode(data_, Lors(lors_data))


@nef_class
class PetEcatListmodeToSinogram:
    scanner: PetEcatScanner

    def __call__(self, listmode: Listmode) -> Sinogram:
        pos1 = listmode.lors.data[:, :3]
        pos2 = listmode.lors.data[:, 3:6]
        irow_ = EcatCrystalPosToIndex(self.scanner)(pos1)
        icol_ = EcatCrystalPosToIndex(self.scanner)(pos2)
        N = self.scanner.nb_crystals
        sino_data = coo_matrix((listmode.data, (irow_, icol_)), shape = (N, N))
        return Sinogram(sino_data, self.scanner)


@nef_class
class PetEcatListmodeTrans:
    scanner: PetEcatScanner

    def __call__(self, listmode: Listmode) -> Listmode:
        pos1 = listmode.lors.data[:, :3]
        pos2 = listmode.lors.data[:, 3:6]
        ind1 = EcatCrystalPosToIndex(self.scanner)(pos1)
        ind2 = EcatCrystalPosToIndex(self.scanner)(pos2)
        pos1_new = EcatIndexToCrystalPos(self.scanner)(ind1)
        pos2_new = EcatIndexToCrystalPos(self.scanner)(ind2)
        if listmode.lors.data.shape[1] == 6:
            lors_data_new = np.hstack((pos1_new, pos2_new)).astype(np.float32)
        else:
            tof_val = listmode.lors.data[:, -1]
            lors_data_new = np.hstack((pos1_new, pos2_new, tof_val.reshape(-1, 1))).astype(
                np.float32)
        return listmode.update(lors = Lors(lors_data_new))
