# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: tof_config.py
@date: 5/15/2019
@desc:
'''
from srfnef import nef_class
import attr


@nef_class
class TofConfig:
    tof_bin: float = attr.ib(default = 3.66)
    tof_sigma2: float = attr.ib(default = 1000)
