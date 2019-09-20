# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: detector_delay.py
@date: 9/2/2019
@desc:
'''
import numpy as np
import srfnef as nef
from srfnef.adapters.gate_single_loader import GateSingleLoader
from scipy import sparse
from scipy.sparse.linalg import lsqr


def max_shift_val(time1, time2, shift_max):
    shift_, val, val_list = 0, 0, []

    for k in range(-shift_max, shift_max + 1):
        val_ = np.intersect1d(time1 + k, time2).size
        val_list.append(val_)
        if val_ > val:
            val = val_
            shift_ = -k
    return shift_, val, val_list


@nef.nef_class
class DetectorDelaySolver:
    scanner: nef.PetCylindricalScanner
    loader: GateSingleLoader

    def __call__(self, filename) -> np.ndarray:
        time_, crystal_id = self.loader(filename.replace('?', '0'))
        detector_id = crystal_id // self.scanner.nb_crystal_per_module
        nb_detector = self.scanner.nb_module[0] * self.scanner.nb_module[1]
        nb_panel = self.scanner.nb_rsector
        unique_detector_id = np.arange(nb_panel * nb_detector)
        row, col, data = np.array([]), np.array([]), np.array([])
        d = np.array([])

        for i1 in nef.utils.tqdm(unique_detector_id):
            for i2 in unique_detector_id:
                if i2 <= i1:
                    continue
                if (i1 // nb_detector - i2 // nb_detector) % nb_panel not in nb_panel // 2 + \
                        np.arange(-1, 2):
                    continue
                time1 = (time_[detector_id == i1] // 1000).astype(np.int64)
                time2 = (time_[detector_id == i2] // 1000).astype(np.int64)
                row = np.hstack((row, [d.size, d.size]))
                col = np.hstack((col, [i1, i2]))
                data = np.hstack(([data, [-1, 1]]))
                ans = max_shift_val(time1, time2, 30)
                d = np.hstack((d, ans[0]))
                # print(i1, i2, ans[0])
        A = sparse.coo_matrix((data, (row.astype(np.uint32), col.astype(np.uint32))))
        return lsqr(A, -d)[0] * 1000
