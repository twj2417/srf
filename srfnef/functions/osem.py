# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: srf_ct
@file: recon_related.py
@date: 4/8/2019
@desc:
'''
import numpy as np
import tensorflow as tf
from srfnef import nef_class
from srfnef.data import Image, Listmode, Emap
from srfnef.utils import tqdm
from srfnef.functions import BackProject, Project

"""
接收以list mode格式的图像信息，根据OSEM算法，可用python的numba库运算，也可以用TensorFlow运算。功能：返回用OSEM加速运算后的重建图像。
"""
@nef_class
class Osem:

    n_iter: int
    n_sub: float
    emap: Emap

    def __call__(self, listmode: Listmode) -> Image:
        if self.mode == 'numba':
            sub_length = int(len(listmode) // self.n_sub)
            x = self.emap.update(data = np.ones(self.emap.shape, dtype = np.float32))
            for _ in tqdm(range(self.n_iter)):
                index = np.random.randint(0, high = len(listmode), size = sub_length)
                _listmode = Project(self.mode)(x, listmode.lors[index, :])
                _bp = BackProject(self.mode)(listmode[index] / _listmode, x)
                x *= _bp / self.emap
            return x
        elif self.mode == 'tf-eager':
            raise NotImplementedError
            x = self.emap.update(data = np.ones(self.emap.shape, dtype = np.float32))
            x_tf = x.update(data = tf.constant(x.data))
            emap_tf = self.emap.update(data = tf.constant(self.emap.data))
            lors_tf = listmode.lors.update(data = tf.constant(listmode.lors.data))
            listmode_tf = listmode.update(data = tf.constant(listmode.data), lors = lors_tf)
            _listmode_tf = Project(self.mode)(x_tf, lors_tf)
            _bp = BackProject(self.mode)(listmode_tf / _listmode_tf, emap_tf)
            _mlem_iter = tf.assign(x_tf.data, (x_tf * _bp / emap_tf).data)

            with tf.compat.v1.Session('') as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                for _ in tqdm(range(self.n_iter)):
                    sess.run(_mlem_iter)
                return x_tf.update(data = sess.run(x_tf.data))
        elif self.mode == 'tf':
            raise NotImplementedError


@nef_class
class Osem2:
    n_iter: int
    n_sub: float
    project: Project
    bproject: BackProject
    emap: Emap
    # saver: Saver
    is_tqdm: bool

    def __call__(self, listmode: Listmode) -> Image:
        return Image()
        # x = Image(np.ones(self.emap._shape, dtype = np.float32), self.emap.center,
        #           self.emap.size)
        #
        # _range = range(self.n_iter) if self.is_tqdm is None else tqdm(range(self.n_iter))
        #
        # sub_length = len(listmode) // self.n_sub
        # index0 = np.arange(len(listmode))
        #
        # for ind in _range:
        #     index = index0[ind * sub_length + np.arange(sub_length)]
        #     pj = self.projector(x, listmode.lors[index, :])
        #     bp = self.back_projector(listmode[index].evolve(lors = listmode.lors[index, :]) / pj,
        #                              self.emap)
        #     x *= bp / self.emap
        # return x
