# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: gate_single_loader.py
@date: 9/2/2019
@desc:
'''
from srfnef import nef_class, PetCylindricalScanner
import deepdish as dd
import numpy as np
import attr


@nef_class
class GateSingleLoader:
    scanner: PetCylindricalScanner
    measure_error: float
    detector_delay: float
    subblock_delay: float
    crystal_delay: float
    is_sorted: bool
    all_true: bool = attr.ib(default = False)
    run_id: int = attr.ib(default = None)

    def __call__(self, filename):

        field_names = [
            '/time', '/rsector_id', '/module_id', '/submodule_id',
            '/crystal_id'
        ]

        nb_all_crystal = self.scanner.nb_all_crystal
        nb_all_panel = self.scanner.nb_rsector
        nb_detector = self.scanner.nb_module[0] * self.scanner.nb_module[1]
        nb_all_detector = nb_all_panel * nb_detector
        nb_subblock = self.scanner.nb_submodule[0] * self.scanner.nb_submodule[1]
        nb_all_subblock = nb_all_detector * nb_subblock
        nb_crystal = self.scanner.nb_crystal[0] * self.scanner.nb_crystal[1]
        if np.isscalar(self.detector_delay):
            object.__setattr__(self, 'detector_delay', np.random.uniform(0, self.detector_delay,
                                                                         nb_all_detector))

        if np.isscalar(self.subblock_delay):
            object.__setattr__(self, 'subblock_delay', np.random.uniform(0, self.subblock_delay,
                                                                         nb_all_subblock))

        if np.isscalar(self.crystal_delay):
            object.__setattr__(self, 'crystal_delay', np.random.uniform(0, self.crystal_delay,
                                                                        nb_all_crystal))

        true_times, panel_id, detector_id, subblock_id, crystal_id = dd.io.load(filename,
                                                                                field_names)
        if self.run_id is not None:
            run_all_id = dd.io.load(filename, '/run_id')
            run_id_filt = run_all_id == self.run_id
            true_times, panel_id, detector_id, subblock_id, crystal_id = true_times[run_id_filt], \
                                                                         panel_id[run_id_filt], \
                                                                         detector_id[run_id_filt], \
                                                                         subblock_id[run_id_filt], \
                                                                         crystal_id[run_id_filt]

        if self.all_true:
            event_id = dd.io.load(filename, '/index')
            if self.run_id is not None:
                event_id = event_id[run_id_filt]
            fst_ind = np.where(event_id[:-1] == event_id[1:])[0]
            snd_ind = fst_ind + 1
            all_ind = np.vstack((fst_ind, snd_ind)).transpose().ravel()
            all_detector_id = panel_id.astype(np.int32) * nb_detector + detector_id.astype(np.int32)
            all_subblock_id = subblock_id.astype(np.int32) + all_detector_id * nb_subblock
            all_crystal_id = (crystal_id + nb_crystal * all_subblock_id).astype(np.int32)
            return true_times[all_ind] // 10, all_crystal_id[all_ind]

        measure_error_ = np.random.normal(0, self.measure_error, true_times.size)
        all_detector_id = panel_id.astype(np.int32) * nb_detector + detector_id.astype(np.int32)
        all_subblock_id = subblock_id.astype(np.int32) + all_detector_id * nb_subblock
        all_crystal_id = (crystal_id + nb_crystal * all_subblock_id).astype(np.int32)
        time_ = true_times // 10 + measure_error_ + self.crystal_delay[all_crystal_id] + \
                self.subblock_delay[all_subblock_id] + self.detector_delay[
                    all_detector_id]
        if self.is_sorted:
            ind_sort = np.argsort(time_)
            return time_[ind_sort], all_crystal_id[ind_sort]
        else:
            return time_, all_crystal_id
