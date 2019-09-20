# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: lors_from_fst_snd.py
@date: 5/8/2019
@desc:
'''
import numpy as np
from srfnef.data import Lors
from srfnef import nef_class


@nef_class
class LorsFromFstSnd:
    def __call__(self, fst: np.ndarray, snd: np.ndarray) -> Lors:
        return Lors(np.hstack((fst, snd)))
