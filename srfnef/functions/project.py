# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: project.py
@date: 4/17/2019
@desc:
'''
import tensorflow as tf
from srfnef import nef_class
from srfnef.data import Listmode, Image, Lors
from srfnef.utils import declare_eager_execution
from srfnef.config import TF_USER_OP_PATH

siddon_module = tf.load_op_library(TF_USER_OP_PATH + '/tf_siddon_module.so')

"""
接收图像信息，通过Tensorflow的python接口提供的tf.load_op_library函数执行编译.so文件，与siddon.cc文件对应。功能：返回list mode格式的图像信息。
"""
@nef_class
class Project:
    mode: str

    def _project_siddon_tf(self, image, lors):
        vproj_data = siddon_module.projection(lors = tf.transpose(lors.data),
                                              image = tf.transpose(image.data),
                                              grid = image.shape,
                                              center = image.center,
                                              size = image.size)
        return vproj_data

    def __call__(self, image: Image, lors: Lors) -> Listmode:
        if self.mode == 'tf-eager':
            declare_eager_execution()
            proj_data = self._project_siddon_tf(image, lors).numpy()
            return Listmode(proj_data, lors)
        elif self.mode == 'tf':
            proj_data = self._project_siddon_tf(image, lors)
            return Listmode(proj_data, lors)
