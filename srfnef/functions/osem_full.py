# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: osem_full.py
@date: 5/21/2019
@desc:
'''
import numpy as np
import tensorflow as tf
from srfnef import nef_class, NoneType
from .data_transform import PetListmodeTrans
from srfnef.utils import tqdm
from srfnef.data import Image, Listmode, Emap
from srfnef.functions import BackProject, Project, EmapGenerator
from srfnef.geometry import PetEcatScanner, ImageConfig
from srfnef.corrections import AttenuationCorrect
from srfnef.corrections.scattering import ScatterCorrect
# from srfnef.corrections import NormalizationAngledCorrect
from srfnef.corrections.psf import PsfCorrect

from srfnef.utils import declare_eager_execution

"""
输入参数指示迭代次数、图像配置、探测器物理信息、存储投影数据的格式、校正信息等，参数根据OSEM算法，基于TensorFlow进行运算。功能：返回重建后的图像。
"""


@nef_class
class OsemFull:
    n_iter: int
    n_sub: int
    image_config: ImageConfig
    scanner: PetEcatScanner
    listmode: Listmode

    atten_corr: AttenuationCorrect
    scatter_corr: ScatterCorrect
    norm_corr: bool
    psf_corr: PsfCorrect
    emap: Emap
    image: Image

    def generate_emap(self) -> NoneType:
        emap = EmapGenerator('ring', self.scanner)(self.image_config)
        object.__setattr__(self, 'emap', emap)

    def __call__(self) -> Image:
        declare_eager_execution()
        if self.emap is None:
            self.generate_emap()

        listmode_ = PetListmodeTrans(self.scanner)(self.listmode)

        # AttenuationCorrect
        if self.atten_corr is not None:
            listmode_ = self.atten_corr(listmode_)

        x_tf = Image(data = tf.Variable(np.ones(self.image_config.shape, dtype = np.float32)),
                     center = self.image_config.center,
                     size = self.image_config.size)
        rand_inds = np.arange(len(self.listmode))
        np.random.shuffle(rand_inds)
        n_per_sub = len(self.listmode) // self.n_sub
        emap_tf = self.emap.update(data = tf.constant(self.emap.data))
        for i_iter in tqdm(range(self.n_iter)):
            ind = i_iter // self.n_sub
            index = rand_inds[ind * n_per_sub: (ind + 1) * n_per_sub]
            lors_tf = self.listmode.lors.update(data = tf.constant(self.listmode.lors.data[index,
                                                                   :]))
            listmode_tf = self.listmode.update(data = tf.constant(listmode_.data[index]),
                                               lors = lors_tf)
            _listmode_tf = Project('tf-eager')(x_tf, lors_tf)
            _listmode_tf = _listmode_tf + 1e-8
            _bp = BackProject('tf-eager')(listmode_tf / _listmode_tf, emap_tf)
            x_tf = x_tf * _bp / emap_tf

        x_tf_data = x_tf.data.numpy()
        x = x_tf.update(data = x_tf_data)
        if self.psf_corr is not None:
            image_ = self.psf_corr(x)
        else:
            image_ = x

        # return image_
        return image_
