# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: crystal_delay.py
@date: 9/2/2019
@desc:
'''
import numpy as np
import srfnef as nef
from srfnef.adapters.gate_single_loader import GateSingleLoader
from scipy import sparse
from scipy.sparse.linalg import lsqr


@nef.nef_class
class CrystalDelaySolver:
    time_window: float
    scanner: nef.PetCylindricalScanner
    loader: GateSingleLoader

    def __call__(self, num: int, filename: str) -> np.ndarray:
        nb_panel = self.scanner.nb_rsector
        N = self.scanner.nb_all_crystal

        data_mat = sparse.coo_matrix((N, N), dtype = np.float32)
        num_mat = sparse.coo_matrix((N, N), dtype = np.uint32)

        for ind in nef.utils.tqdm(range(num)):
            time_, crystal_id = self.loader(filename.replace('?', str(ind)))
            all_panel_id = crystal_id // self.scanner.nb_crystal_per_rsector
            condition_window = time_[1:] - time_[:-1] < self.time_window
            condition_front = np.isin((all_panel_id[1:] - all_panel_id[:-1]) % nb_panel,
                                      nb_panel // 2 + np.arange(-1, 2))
            fst_ind = np.where(
                np.logical_and(condition_front, condition_window))[0]
            snd_ind = fst_ind + 1
            ind1 = crystal_id[fst_ind]
            ind2 = crystal_id[snd_ind]
            inv = ind1 > ind2

            row_all = ind1 * (1 - inv) + ind2 * inv
            col_all = ind1 * inv + ind2 * (1 - inv)
            dtime_all = (time_[snd_ind] - time_[fst_ind]) * 2 * (0.5 - inv)

            data_mat += sparse.coo_matrix((dtime_all, (row_all, col_all)),
                                          (N, N),
                                          dtype = np.float32)
            num_mat += sparse.coo_matrix(
                (np.ones(dtime_all.size), (row_all, col_all)),
                shape = (N, N),
                dtype = np.uint32)

        data_mat.sum_duplicates()
        num_mat.sum_duplicates()
        new_row_all = np.hstack(
            (np.arange(num_mat.nnz), np.arange(num_mat.nnz)))
        new_col_all = np.hstack((num_mat.nonzero()[0], num_mat.nonzero()[1]))
        new_data_all = np.hstack(
            (np.ones(num_mat.nnz) * -1, np.ones(num_mat.nnz)))
        A = sparse.coo_matrix((new_data_all, (new_row_all, new_col_all)),
                              shape = (num_mat.nnz, N))
        # ans = data_mat.data / num_mat.data
        d = np.nan_to_num(np.divide(data_mat, num_mat))[num_mat.nonzero()]
        return lsqr(A, np.array(d)[0])[0]
