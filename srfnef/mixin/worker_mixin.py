import tensorflow as tf
from dxl.learn import Graph
from dxl.learn.function import dependencies, merge_ops
from srfnef.base.barrier_single import no_op
from srfnef import BackProject, Project
from dxl.learn.tensor import initializer, variable_from_tensor
import matplotlib.pyplot as plt


class WorkerGraph(Graph):
    class KEYS(Graph.KEYS):
        class CONFIG(Graph.KEYS.CONFIG):
            TASK_INDEX = 'task_index'

        class TENSOR(Graph.KEYS.TENSOR):
            X = 'x'
            RESULT = 'result'
            TARGET = 'target'
            INIT = 'init'
            UPDATE = 'update'
            EFFICIENCY_MAP = 'efficiency_map'
            PROJECTION_DATA = 'projection_data'

        class GRAPH(Graph.KEYS.GRAPH):
            RECONSTRUCTION = 'reconstruction'

    def __init__(self, name, x = None, x_target = None, task_index = None, inputs = None):
        super().__init__(name)
        self.tensors[self.KEYS.TENSOR.X] = x
        self.tensors[self.KEYS.CONFIG.TASK_INDEX] = task_index
        self.tensors[self.KEYS.TENSOR.TARGET] = x_target
        self.inputs = inputs


    def kernel(self):
        self._construct_inputs()
        self._construct_x_result()
        self._construct_x_update()

    @property
    def task_index(self):
        return self.config[self.KEYS.CONFIG.TASK_INDEX]

    def _construct_inputs(self):
        emap = self.inputs[self.KEYS.TENSOR.EFFICIENCY_MAP]
        emap_tf = emap.update(data = tf.constant(emap.data))
        projection_data = self.inputs[self.KEYS.TENSOR.PROJECTION_DATA]
        lors_tf = projection_data.lors.update(data = tf.constant(projection_data.lors.data))
        projection_data_tf = projection_data.update(data = tf.constant(projection_data.data),
                                                    lors = lors_tf)
        self.tensors[self.KEYS.TENSOR.EFFICIENCY_MAP] = emap_tf
        self.tensors[self.KEYS.TENSOR.PROJECTION_DATA] = projection_data_tf
        with tf.control_dependencies([self.tensors[self.KEYS.TENSOR.EFFICIENCY_MAP].data,
                                      self.tensors[self.KEYS.TENSOR.PROJECTION_DATA].data]):
            self.tensors[self.KEYS.TENSOR.INIT] = tf.no_op()

    def _construct_x_result(self):
        self.tensors[self.KEYS.TENSOR.RESULT] = self.work_step()

    def _construct_x_update(self):
        """
        update the master x buffer with the x_result of workers.
        """
        KT = self.KEYS.TENSOR
        self.tensors[KT.UPDATE] = tf.compat.v1.assign(self.tensors[KT.TARGET],
                                                      self.tensors[KT.RESULT].data)
        # with tf.control_dependencies([self.tensors[KT.TARGET]]):
        #     self.tensors[KT.UPDATE] = tf.no_op()

    def work_step(self):
        image = self.tensors[self.KEYS.TENSOR.X]
        efficiency_map = self.tensors[self.KEYS.TENSOR.EFFICIENCY_MAP]
        projection_data = self.tensors[self.KEYS.TENSOR.PROJECTION_DATA]
        proj = Project('tf')(image, projection_data.lors)
        back_proj = BackProject('tf')(projection_data / proj, efficiency_map)
        return image * back_proj / efficiency_map
