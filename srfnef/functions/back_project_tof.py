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
import numpy as np
from srfnef.data import Listmode, Image
from srfnef.utils import declare_eager_execution
from srfnef.config import TF_USER_OP_PATH

siddon_tof_module = tf.load_op_library(TF_USER_OP_PATH + '/tf_siddon_tof_module.so')

"""
接收图像信息，通过Tensorflow的python接口提供的tf.load_op_library函数执行编译.so文件，与siddon.cc文件对应。功能：返回基于飞行时间技术的反投影图像。
"""
class BackProjectTof:
    def __init__(self, mode = None, tof = None):
        self.mode = mode
        self.tof = tof

    def _back_project_siddon_tof_tf(self, listmode, image):
        image_data = siddon_tof_module.backprojection_tof(image = tf.transpose(image.data),
                                                          lors = tf.transpose(listmode.lors.data),
                                                          lors_value = listmode.data,
                                                          grid = image.shape,
                                                          center = image.center,
                                                          size = image.size,
                                                          tof_values = listmode.lors.tof_values,
                                                          tof_bin = self.tof.tof_bin,
                                                          tof_sigma2 = self.tof.tof_sigma2)
        return tf.transpose(image_data)

    def __call__(self, listmode: Listmode, image: Image) -> Image:
        if self.mode == 'tf-eager':
            declare_eager_execution()
            lors_tf = listmode.lors.update(data = tf.constant(listmode.lors.data))
            listmode_tf = listmode.update(data = tf.constant(listmode.data), lors = lors_tf)
            image_tf = Image(tf.constant(np.zeros(image.shape, dtype = np.float32)), image.center,
                             image.size)
            _image_data = self._back_project_siddon_tof_tf(listmode_tf, image_tf).numpy()
            return image.update(data = _image_data)
        elif self.mode == 'tf':
            _image_data = self._back_project_siddon_tof_tf(listmode, image)

            return image.update(data = _image_data)
        else:
            raise NotImplementedError
