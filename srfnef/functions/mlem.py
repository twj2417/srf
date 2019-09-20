# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: srf_ct
@file: app_recon_full.py
@date: 4/8/2019
@desc:
'''

import numpy as np
import tensorflow as tf
from srfnef import nef_class
from srfnef.utils import tqdm
from srfnef.data import Image, Listmode, Emap
from srfnef.functions import BackProject, Project
from srfnef.utils import declare_eager_execution
from copy import copy

"""
给定迭代次数和运算的模型，根据MLEM算法，可用python的numba库运算，也可以用TensorFlow运算。功能：返回重建后的图像。
"""
@nef_class
class Mlem:
    n_iter: int
    emap: Emap

    def __call__(self, listmode: Listmode) -> Image:
        declare_eager_execution()
        x_tf = Image(data = tf.Variable(np.ones(self.emap.shape, dtype = np.float32)),
                     center = self.emap.center,
                     size = self.emap.size)
        emap_data_n0_zero = copy(self.emap.data)
        emap_data_n0_zero[emap_data_n0_zero == 0.0] = 1e8
        emap_tf = self.emap.update(data = tf.constant(emap_data_n0_zero))
        lors_tf = listmode.lors.update(data = tf.constant(listmode.lors.data))
        listmode_tf = listmode.update(data = tf.constant(listmode.data), lors = lors_tf)

        for _ in tqdm(range(self.n_iter)):
            _listmode_tf = Project('tf-eager')(x_tf, lors_tf)
            _listmode_tf = _listmode_tf + np.mean(listmode.data) * 1e-8
            _bp = BackProject('tf-eager')(listmode_tf / _listmode_tf, emap_tf)
            x_tf = x_tf * _bp / emap_tf

        return x_tf.update(data = x_tf.data.numpy())
