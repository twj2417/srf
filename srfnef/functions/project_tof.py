# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: project_tof.py
@date: 4/17/2019
@desc:
'''
import tensorflow as tf
from srfnef import nef_class
from srfnef.data import Listmode, Image, Lors
from srfnef.geometry import TofConfig
from srfnef.utils import declare_eager_execution
from srfnef.config import TF_USER_OP_PATH

siddon_tof_module = tf.load_op_library(TF_USER_OP_PATH + '/tf_siddon_tof_module.so')

"""
接收图像信息和飞行时间技术参数，通过Tensorflow的python接口提供的tf.load_op_library函数执行编译.so文件，与siddon.cc文件对应。功能：返回list mode格式的图像信息。
"""
@nef_class
class ProjectTof:
    mode: str
    tof: TofConfig

    def _project_siddon_tof_tf(self, image, lors):
        vproj_data = siddon_tof_module.projection_tof(lors = tf.transpose(lors.data),
                                                      image = tf.transpose(image.data),
                                                      grid = image.shape,
                                                      center = image.center,
                                                      size = image.size,
                                                      tof_values = lors.tof_values,
                                                      tof_bin = self.tof.tof_bin,
                                                      tof_sigma2 = self.tof.tof_sigma2)
        return vproj_data

    def __call__(self, image: Image, lors: Lors) -> Listmode:
        if self.mode == 'tf-eager':
            declare_eager_execution()
            proj_data = self._project_siddon_tof_tf(image, lors).numpy()
            return Listmode(proj_data, lors)
        elif self.mode == 'tf':
            proj_data = self._project_siddon_tof_tf(image, lors)
            return Listmode(proj_data, lors)
        else:
            raise NotImplementedError
