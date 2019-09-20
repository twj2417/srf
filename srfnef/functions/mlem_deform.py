# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: srf_ct
@file: mlem_deform.py
@date: 4/8/2019
@desc:
'''

import numpy as np
import tensorflow as tf
from srfnef import nef_class, List
from srfnef.utils import tqdm
from srfnef.data import Image, Listmode, Emap
from srfnef.functions import BackProject, Project
from srfnef.ops.deform_mixins import DeformMixin
from srfnef.data.dvf import Dvf
from srfnef.utils import declare_eager_execution


@nef_class
class MlemDeform(DeformMixin):
    n_iter: int
    emap: Emap

    def __call__(self, listmodes: List(Listmode), dvfs: List(Dvf)) -> Image:
        declare_eager_execution()
        x = self.emap.update(data = np.ones(self.emap.shape, dtype = np.float32))
        x_tf = x.update(data = tf.Variable(x.data))

        emap_tf = self.emap.update(data = tf.constant(self.emap.data))
        for _ in tqdm(range(self.n_iter)):
            for listmode, dvf in zip(listmodes, dvfs):
                dvf_x_tf = tf.constant(dvf.dvf_x.data)
                dvf_y_tf = tf.constant(dvf.dvf_y.data)
                dvf_z_tf = tf.constant(dvf.dvf_z.data)
                lors_tf = listmode.lors.update(data = tf.constant(listmode.lors.data))
                listmode_tf = listmode.update(data = tf.constant(listmode.data), lors = lors_tf)
                _x_tf = x_tf.update(
                    data = self._deform_invert_tf(x_tf.data, dvf_x_tf, dvf_y_tf, dvf_z_tf))
                _listmode_tf = Project('tf')(_x_tf, lors_tf)
                _listmode_tf = _listmode_tf + 1e-8
                _bp = BackProject('tf')(listmode_tf / _listmode_tf, emap_tf)
                emap_tf2 = emap_tf.update(
                    data = self._deform_tf(emap_tf.data, dvf_x_tf, dvf_y_tf, dvf_z_tf)) + 1e-8
                _bp2 = _bp.update(data = self._deform_tf(_bp.data, dvf_x_tf, dvf_y_tf, dvf_z_tf))
                x_tf = x_tf * _bp2 / emap_tf2

        return x_tf.update(data = x_tf.data.numpy())
