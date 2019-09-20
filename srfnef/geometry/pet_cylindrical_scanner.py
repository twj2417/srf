# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: pet_cylindrical_scanner.py
@date: 6/11/2019
@desc:
'''
from numpy import pi
from srfnef import nef_class, List
from .tof_config import TofConfig
from srfnef.geometry import PetScanner
from srfnef.ops.geometry.pet_cylindrical_scanner import PropertyClass

'''
One layer Cylindrical PET system
- rsector (rotational sector)
- module
- submodule
- crystal

Descriptions:
- the rsector arrangement is in right-handed Cartesian (anti-clockwise numbering) coordinate 
system, start from x-positive
- furthermore can be found in 
http://www.castor-project.org/sites/default/files/2018-11/CASToR_general_documentation.pdf
Section 5
'''


@nef_class
class PetCylindricalScanner(PetScanner, PropertyClass):
    inner_radius: float
    outer_radius: float
    nb_rsector: int

    nb_module: List(int, 2)  # module grid in each rsector, [trans, z]
    mv_module: List(float, 2)  # movement per module

    nb_submodule: List(int, 2)  # submodule grid in each module, [trans, z]
    mv_submodule: List(float, 2)  # movement per submodule

    nb_crystal: List(int, 2)  # crystal grid in each submodule, [trans, z]
    mv_crystal: List(float, 2)  # movement per crystal

    sz_crystal: List(float, 2)  # crystaml voxel-size

    tof: TofConfig
