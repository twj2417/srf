import attr
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from dxl.learn.graph.master_worker_task import MasterWorkerTaskBase
from srfnef import nef_class
from srfnef import Listmode, Emap, Lors
from srfnef.base.barrier_single import barrier_single
from srfnef.mixin.master_mixin import MasterGraph
from srfnef.mixin.worker_mixin import WorkerGraph
from dxl.learn.distribute import ThisHost, make_master_worker_cluster
from dxl.learn.core import ThisSession
from srfnef.io import load
from srfnef.utils import config_with_name
import matplotlib.pyplot as plt


class Distribute(MasterWorkerTaskBase):
    # info:str = attr.ib(default = 'distribute_reconstruction_task')
    # job:str = attr.ib(default = 'master')
    # task_index:int = attr.ib(default = 0)
    # cluster_config:dict = attr.ib(default = {})
    # task_config:dict = attr.ib(default = {})
    # graphs:MasterGraph = attr.ib(default = None)

    class KEYS(MasterWorkerTaskBase.KEYS):
        class CONFIG(MasterWorkerTaskBase.KEYS.CONFIG):
            NB_ITERATIONS = 'nb_iterations'
            EMAP_PATH = 'emap_path'
            PROJ_PATH = 'proj_path'

        class TENSOR(MasterWorkerTaskBase.KEYS.TENSOR):
            X = 'x'
            PROJECTION_DATA = 'projection_data'
            EFFICIENCY_MAP = 'efficiency_map'
            INIT = 'init'
            RECON = 'recon'
            MERGE = 'merge'

        class GRAPH(MasterWorkerTaskBase.KEYS.GRAPH):
            MASTER = 'master'
            WORKER = 'worker'

    def __init__(self, info = 'distribute_reconstruction_task', job = 'master', task_index = 0,
                 cluster_config = None, task_config = None):
        self.config = config_with_name(info)
        # cluster_spec = MasterWorkerClusterSpec(cluster_config)
        cluster = make_master_worker_cluster(cluster_config, job, task_index)
        super().__init__(info = info, config = task_config, tensors = info, job = job,
                         task_index = task_index, cluster = cluster)
        self.config.update(self.KEYS.CONFIG.NB_ITERATIONS, task_config['nb_iteration'])
        self.config.update(self.KEYS.CONFIG.EMAP_PATH, task_config['emap'])
        self.config.update(self.KEYS.CONFIG.PROJ_PATH, task_config['listmode'])

    def kernel(self):
        self._make_master_graph()
        self._make_worker_graph()
        self._make_barriers()

    def _load_data(self, key):
        KC = self.KEYS.CONFIG
        if key == KC.EMAP_PATH:
            result = load(Emap, self.config.get(key))
        else:
            listmode = load(Listmode, self.config.get(key))
            len_per_sub = listmode.data.shape[0] // self.nb_workers
            tid = self.task_index
            result = Listmode(listmode.data[tid * len_per_sub:(tid + 1) * len_per_sub],
                              Lors(
                                  listmode.lors.data[tid * len_per_sub:(tid + 1) * len_per_sub, :]))
        return result

    def _make_master_graph(self):
        emap = load(Emap, self.config.get(self.KEYS.CONFIG.EMAP_PATH))
        m = MasterGraph(self.nb_workers, emap, self.name + 'master')
        m.make()
        self.graphs[self.KEYS.GRAPH.MASTER] = m
        self.tensors[self.KEYS.TENSOR.X] = m.tensors[self.KEYS.TENSOR.X]

    def _make_worker_graph(self):
        KS, KT, KC = self.KEYS.GRAPH, self.KEYS.TENSOR, self.KEYS.CONFIG
        if not ThisHost.is_master():
            self.graphs[KS.WORKER] = [
                None for i in range(self.nb_workers)]
            m = self.graphs[KS.MASTER]
            name = self.name + 'worker_{}'.format(self.task_index)
            inputs = {
                KT.EFFICIENCY_MAP: self._load_data(KC.EMAP_PATH),
                KT.PROJECTION_DATA: self._load_data(KC.PROJ_PATH)
            }
            w = WorkerGraph(name, m.tensors[m.KEYS.TENSOR.X],
                            m.tensors[m.KEYS.TENSOR.BUFFER][self.task_index], self.task_index,
                            inputs)
            w.make()
            self.graphs[KS.WORKER][self.task_index] = w

    def _make_init_barrier(self):
        mg = self.graphs[self.KEYS.GRAPH.MASTER]
        name = self.name + "barrier_{}".format(self.KEYS.TENSOR.INIT)
        if ThisHost.is_master():
            task = mg.tensors[mg.KEYS.TENSOR.INIT]
            id_join = self.nb_workers
        else:
            wg = self.graphs[self.KEYS.GRAPH.WORKER][self.task_index]
            task = wg.tensors[wg.KEYS.TENSOR.INIT]
            id_join = self.task_index
        init_op = barrier_single(name, 1 + self.nb_workers, 1 + self.nb_workers,
                                 task, id_join)
        self.tensors[self.KEYS.TENSOR.INIT] = init_op

    def _make_recon_barrier(self):
        mg = self.graphs[self.KEYS.GRAPH.MASTER]
        name = self.name + "barrier_{}".format(self.KEYS.TENSOR.RECON)
        if ThisHost.is_master():
            task = None
            id_join = 0
        else:
            wg = self.graphs[self.KEYS.GRAPH.WORKER][self.task_index]
            task = wg.tensors[wg.KEYS.TENSOR.UPDATE]
            id_join = None
        recon_op = barrier_single(name, self.nb_workers, 1, task, id_join)
        self.tensors[self.KEYS.TENSOR.RECON] = recon_op

    def _make_merge_barrier(self):
        mg = self.graphs[self.KEYS.GRAPH.MASTER]
        name = self.name + "barrier_{}".format(self.KEYS.TENSOR.MERGE)
        if ThisHost.is_master():
            task = mg.tensors[mg.KEYS.TENSOR.UPDATE]
            id_join = None
        else:
            wg = self.graphs[self.KEYS.GRAPH.WORKER][self.task_index]
            task = None
            id_join = self.task_index
        merge_op = barrier_single(name, 1, self.nb_workers, task, id_join)
        self.tensors[self.KEYS.TENSOR.MERGE] = merge_op

    def _make_barriers(self):
        self._make_init_barrier()
        self._make_recon_barrier()
        self._make_merge_barrier()

    def run_task(self):
        KT = self.KEYS.TENSOR
        KC = self.KEYS.CONFIG
        ThisSession.run(self.tensors[KT.INIT])
        for i in tqdm(range(self.config[KC.NB_ITERATIONS])):
            ThisSession.run(self.tensors[KT.RECON])
            ThisSession.run(self.tensors[KT.MERGE])
            if ThisHost.is_master():
                x = ThisSession.run(self.tensors[KT.X].data)
                np.save(f'./result_{i}.npy', x)
