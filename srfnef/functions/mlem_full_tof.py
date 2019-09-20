# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: mlem_full.py
@date: 5/10/2019
@desc:
'''
import numpy as np
import tensorflow as tf
from srfnef import nef_class, NoneType
from srfnef.utils import tqdm
from srfnef.data import Image, Listmode, Emap
from srfnef.functions import BackProjectTof, ProjectTof, EmapGenerator
from srfnef.geometry import PetEcatScanner, ImageConfig
from srfnef.corrections import AttenuationCorrect
from srfnef.corrections.scattering import ScatterCorrect
from srfnef.corrections.psf import PsfCorrect
from srfnef.utils import declare_eager_execution
from copy import copy

"""
输入参数指示迭代次数、图像配置、探测器物理信息、存储投影数据的格式、校正信息等，参数根据TOF MLEM算法，基于TensorFlow进行运算。功能：返回重建的图像。
"""


@nef_class
class MlemFullTof:
    n_iter: int
    image_config: ImageConfig
    scanner: PetEcatScanner
    listmode: Listmode
    atten_corr: AttenuationCorrect
    scatter_corr: ScatterCorrect
    psf_corr: PsfCorrect
    emap: Emap

    def generate_emap(self) -> NoneType:
        emap = EmapGenerator('block-full', self.scanner)(self.image_config)
        object.__setattr__(self, 'emap', emap)

    def __call__(self) -> Image:
        declare_eager_execution()
        if self.emap is None:
            self.generate_emap()

        if self.atten_corr is not None:
            listmode_ = self.atten_corr(self.listmode)
        else:
            listmode_ = self.listmode

        x_tf = Image(data = tf.Variable(np.ones(self.image_config.shape, dtype = np.float32)),
                     center = self.image_config.center,
                     size = self.image_config.size)
        emap_data_n0_zero = copy(self.emap.data)
        emap_data_n0_zero[emap_data_n0_zero == 0.0] = 1e8
        emap_tf = self.emap.update(data = tf.constant(emap_data_n0_zero))
        lors_tf = self.listmode.lors.update(data = tf.constant(self.listmode.lors.data))
        listmode_tf = self.listmode.update(data = tf.constant(listmode_.data), lors = lors_tf)

        for _ in tqdm(range(self.n_iter)):
            _listmode_tf = ProjectTof('tf-eager', self.scanner.tof)(x_tf, lors_tf)
            listmode_div = tf.div_no_nan(listmode_tf.data, _listmode_tf.data + 0.00001)
            _bp = BackProjectTof('tf-eager', self.scanner.tof)(
                listmode_tf.update(data = listmode_div), emap_tf)
            x_tf = x_tf * _bp / emap_tf

        x = x_tf.update(data = x_tf.data.numpy())

        if self.scatter_corr is not None:
            assert self.atten_corr is not None
            listmode_ = self.scatter_corr(x, self.atten_corr.u_map, self.scanner, self.listmode)

            x_tf = Image(data = tf.Variable(np.ones(self.image_config.shape, dtype = np.float32)),
                         center = self.image_config.center,
                         size = self.image_config.size)
            emap_data_n0_zero = copy(self.emap.data)
            emap_data_n0_zero[emap_data_n0_zero == 0.0] = 1e8
            emap_tf = self.emap.update(data = tf.constant(emap_data_n0_zero))
            lors_tf = self.listmode.lors.update(data = tf.constant(self.listmode.lors.data))
            listmode_tf = self.listmode.update(data = tf.constant(listmode_.data), lors = lors_tf)

            for _ in tqdm(range(self.n_iter)):
                _listmode_tf = ProjectTof('tf-eager', self.scanner.tof)(x_tf, lors_tf)
                listmode_div = tf.div_no_nan(listmode_tf.data, _listmode_tf.data + 0.00001)
                _bp = BackProjectTof('tf-eager', self.scanner.tof)(listmode_tf.update(data =
                                                                                      listmode_div),
                                                                   emap_tf)
                x_tf = x_tf * _bp / emap_tf

            x = x_tf.update(data = x_tf.data.numpy())

        if self.psf_corr is not None:
            image_ = self.psf_corr(x)
        else:
            image_ = x

        return image_
