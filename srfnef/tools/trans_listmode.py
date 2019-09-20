# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: trans_listmode.py
@date: 5/13/2019
@desc:
'''

import numpy as np

from srfnef.functions import ListmodeCompress
from srfnef.data import Lors, Listmode
from srfnef.geometry import PetEcatScanner


def trans_listmode(path: str, scanner: PetEcatScanner) -> Listmode:
    ''' to translate gate output lors data to trans listmode data'''
    lors_data = np.load(path)
    if lors_data.shape[1] > 6:
        print('the width of lors data is', lors_data.shape[1], ', take care')
    lors = Lors(lors_data[:, :6])
    listmode = Listmode(np.ones((lors.length,), dtype = np.float32), lors)
    listmode1 = ListmodeCompress(scanner)(listmode)
    return listmode1
