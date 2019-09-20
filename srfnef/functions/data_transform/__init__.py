# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: __init__.py
@date: 9/18/2019
@desc:
'''
from .pet_ecat import PetEcatListmodeToSinogram, PetEcatListmodeTrans, PetEcatScanner, \
    PetEcatSinogramToListmode
from .pet_cylindrical import PetCylindricalSinogramToListmode, PetCylindricalListmodeTrans, \
    PetCylindricalListmodeToSinogram, PetCylindricalScanner
from srfnef import nef_class
from srfnef.geometry import PetScanner


@nef_class
class PetListmodeToSinogram:
    scanner: PetScanner

    def __call__(self, listmode):
        if isinstance(self.scanner, PetEcatScanner):
            return PetEcatListmodeToSinogram(self.scanner)(listmode)
        if isinstance(self.scanner, PetCylindricalScanner):
            return PetCylindricalListmodeToSinogram(self.scanner)(listmode)


@nef_class
class PetSinogramToListmode:
    scanner: PetScanner

    def __call__(self, sinogram):
        if isinstance(self.scanner, PetEcatScanner):
            return PetEcatSinogramToListmode(self.scanner)(sinogram)
        if isinstance(self.scanner, PetCylindricalScanner):
            return PetCylindricalSinogramToListmode(self.scanner)(sinogram)


@nef_class
class PetListmodeTrans:
    scanner: PetScanner

    def __call__(self, listmode):
        if isinstance(self.scanner, PetEcatScanner):
            return PetEcatListmodeTrans(self.scanner)(listmode)
        if isinstance(self.scanner, PetCylindricalScanner):
            return PetCylindricalListmodeTrans(self.scanner)(listmode)
