import attr
import tensorflow as tf
import numpy as np
from srfnef import Emap, Image

from dxl.learn.tensor import no_op, variable_from_tensor, variable, initializer, sum_, assign
from dxl.learn import Graph
from dxl.learn.function import dependencies, merge_ops
from srfnef.utils import config_with_name


class MasterGraph(Graph):
    class KEYS(Graph.KEYS):
        class CONFIG(Graph.KEYS.CONFIG):
            NB_WORKERS = 'nb_workers'
            RENORMALIZATION = 'renormalization'

        class TENSOR(Graph.KEYS.TENSOR):
            X = 'x'
            BUFFER = 'x_buffer'
            UPDATE = 'x_update'
            INIT = 'init'

    def __init__(self, nb_workers, emap = None, name = 'master', is_renormalization = False):
        self.config = config_with_name(name)
        super().__init__(name)
        self.emap = emap
        self.config.update(self.KEYS.CONFIG.NB_WORKERS, nb_workers)
        self.config.update_value_and_default(self.KEYS.CONFIG.RENORMALIZATION, is_renormalization,
                                             False)

    def kernel(self):
        self._construct_x()
        self._construct_init()
        self._construct_summation()

    @property
    def nb_workers(self):
        return self.config.get(self.KEYS.CONFIG.NB_WORKERS)

    def _construct_x(self):
        # x = variable_from_tensor[tf](np.ones_like(self.emap.data), self.KEYS.TENSOR.X)
        x = Emap(np.ones_like(self.emap.data), self.emap.center, self.emap.size)
        self.tensors[self.KEYS.TENSOR.BUFFER] = [
            variable[tf](shape = x.data.shape,
                         dtype = x.data.dtype,
                         name = f'{self.KEYS.TENSOR.BUFFER}_{i}')
            for i in range(self.config.get(self.KEYS.CONFIG.NB_WORKERS))
        ]
        self.tensors[self.KEYS.TENSOR.X] = x.update(data = tf.Variable(x.data))

    def _construct_init(self):
        KT = self.KEYS.TENSOR
        with tf.control_dependencies([self.tensors[KT.X].data]):
            self.tensors[self.KEYS.TENSOR.INIT] = tf.no_op()
        # to_init = [self.tensors[self.KEYS.TENSOR.X].data] + self.tensors[self.KEYS.TENSOR.BUFFER]
        # self.tensors[self.KEYS.TENSOR.INIT] = merge_ops([initializer(t) for t in to_init])

    def _construct_summation(self):
        KT = self.KEYS.TENSOR
        x_s = sum_(self.tensors[KT.BUFFER], axis = 0)
        if self.config[self.KEYS.CONFIG.RENORMALIZATION]:
            x_s = x_s / sum_(x_s) * sum_(self.tensors[KT.X].data)
        x_u = self.tensors[KT.X].update(data = tf.compat.v1.assign(self.tensors[KT.X].data, x_s))
        with tf.control_dependencies([x_u.data]):
            self.tensors[KT.UPDATE] = tf.no_op()
        return x_u
        # self.tensors[KT.UPDATE] = self.tensors[KT.X].data.assign(x_s)
