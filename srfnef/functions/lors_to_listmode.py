# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: scanner_to_lors.py
@date: 4/26/2019
@desc:
'''
import numpy as np
from srfnef import nef_class
from srfnef.data import Lors, Listmode


@nef_class
class LorsToListmode:
    def __call__(self, lors: Lors) -> Listmode:
        _listmode_data = np.ones((lors.shape[0],), dtype = np.float32)
        return Listmode(_listmode_data, lors)
