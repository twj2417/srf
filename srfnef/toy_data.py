# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: toy_data.py
@date: 5/8/2019
@desc:
'''
import srfnef as nef
import numpy as np

block = nef.Block([20.0, 33.4, 33.4], [1, 10, 10])
scanner = nef.PetEcatScanner(99.0, 119.0, 1, 16, 0.0, block)
shape = [90, 90, 10]
center = [0., 0., 0.]
size = [180., 180., 33.4]

lors = nef.Lors(np.load(nef.config.DATABASE_DIR + '/toy_cases/lors.npy'))

listmode = nef.LorsToListmode()(lors) * 0.0

image = nef.Image(np.zeros(shape, dtype = np.float32), center, size)
