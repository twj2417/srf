# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: pet_scanner.py
@date: 6/28/2019
@desc:
'''
from abc import abstractmethod


class PetScanner:
    @abstractmethod
    def crystal_pos_to_ind(self, *args, **kwargs):
        pass

    @abstractmethod
    def ind_to_crystal_pos(self, *args, **kwargs):
        pass
