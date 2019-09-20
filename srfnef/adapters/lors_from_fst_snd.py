# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: lors_from_fst_snd.py
@date: 4/30/2019
@desc:
'''
from srfnef.data_classes import Lors
import numpy as np


def lors_from_fst_snd(fst: np.ndarray, snd: np.ndarray) -> Lors:
    lors_data = np.hstack((fst, snd))
    return Lors(lors_data)
