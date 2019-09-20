import tensorflow as tf
from srfnef.geometry import TF_USER_OP_PATH

siddon_module = tf.load_op_library(TF_USER_OP_PATH + '/tf_siddon_module.so')


class BackProjectMixin:
    def _back_project_siddon_tf(self, listmode, image):
        image_data = siddon_module.backprojection(image = tf.transpose(image.data),
                                                  lors = tf.transpose(listmode.lors.data),
                                                  lors_value = listmode.data,
                                                  grid = image.shape,
                                                  center = image.center,
                                                  size = image.size)
        return tf.transpose(image_data)
