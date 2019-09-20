# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: project_mixin.py
@date: 4/17/2019
@desc:
'''

import tensorflow as tf
from srfnef.geometry import TF_USER_OP_PATH

siddon_module = tf.load_op_library(TF_USER_OP_PATH + '/tf_siddon_module.so')


class ProjectMixin:
    def _project_siddon_tf(self, image, lors):
        vproj_data = siddon_module.projection(lors = tf.transpose(lors.data),
                                              image = tf.transpose(image.data),
                                              grid = image.shape,
                                              center = image.center,
                                              size = image.size)
        return vproj_data
