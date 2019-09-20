# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: loader.py
@date: 9/2/2019
@desc:
'''
import numpy as np
import deepdish as dd
import srfnef as nef
from scipy import sparse
import scipy.fftpack as fftpack

fft = fftpack.fft
ifft = fftpack.ifft


def max_shift_val(sgn1, sgn2, shift_max):
    shift_, val = 0, 0
    for k in range(-shift_max, shift_max + 1):
        if k > 0:
            sum_ = np.sum(sgn1[k:] * sgn2[:-k])
            if sum_ > val:
                val = sum_
                shift_ = k
        elif k == 0:
            sum_ = np.sum(sgn1 * sgn2)
            if sum_ > val:
                val = sum_
                shift_ = k
        elif k < 0:
            sum_ = np.sum(sgn1[:k] * sgn2[-k:])
            if sum_ > val:
                val = sum_
                shift_ = k
    return shift_, val


@nef.nef_class
class SuperSolver:
    config: dict
    scanner: nef.PetCylindricalScanner

    def __call__(self, path: str) -> object:
        nb_submodule = self.scanner.nb_submodule[0] * \
                       self.scanner.nb_submodule[1]
        nb_module = self.scanner.nb_module[0] * self.scanner.nb_module[1]

        if 'num_data' not in self.config:
            raise ValueError('cannot find num_data field in config')
        else:
            pass

        for ind in nef.utils.tqdm(range(self.config['num_data'])):
            filename = path.replace('?', str(ind))
            time_true_np, rsector_id_np, module_id_np, submodule_id_np, crystal_id_np = dd.io.load(
                filename, [
                    '/time', '/rsector_id', '/module_id', '/submodule_id',
                    '/crystal_id'
                ])

            if 'sigma' in self.config:
                sigma = self.config['sigma']
                time_np = time_true_np + \
                          np.random.normal(0, sigma, time_true_np.size)
            else:
                time_np = time_true_np

            all_sub_np = submodule_id_np.astype(
                np.uint32) + nb_submodule * (module_id_np.astype(
                np.uint32) + nb_module * rsector_id_np.astype(np.uint32))
            all_det_id = module_id_np.astype(
                np.uint32) + nb_module * rsector_id_np.astype(np.uint32)

            all_id_np = crystal_id_np + self.scanner.nb_crystal[
                0] * self.scanner.nb_crystal[1] * all_sub_np

            if 'delay_super' in self.config:
                time_np += self.config['delay_super'][all_det_id] * 10000

            if 'estimated_delay_super' in self.config:
                time_np -= self.config['estimated_delay_super'][all_det_id] * 10000

            if 'delay_submodule' in self.config:
                time_np += self.config['delay_submodule'][all_sub_np]

            if 'estimated_delay_submodule' in self.config:
                time_np -= self.config['estimated_delay_submodule'][all_sub_np]

            if 'delay_crystals' in self.config:
                time_np += self.config['delay_crystals'][all_id_np]

            if 'estimated_crystal_delay' in self.config:
                time_np -= self.config['estimated_crystal_delay'][all_id_np]

            sort_ind = np.argsort(time_np)
            time_np = (time_np[sort_ind] / 10000).astype(np.int32)
            all_sub_np = all_sub_np[sort_ind]
            all_det_id = all_det_id[sort_ind]
            N = 10 ** 7

            # A = np.zeros((32, 32))
            row, col, data = np.array([]), np.array([]), np.array([])
            d = np.array([])
            for i1 in nef.utils.tqdm(range(32)):
                for i2 in range(i1, 32):
                    if (i1 // 4 - i2 // 4) % 8 not in [3, 4, 5]:
                        continue
                    time1 = time_np[all_det_id == i1]
                    time1 = time1[time1 < N]
                    sgn1 = sparse.coo_matrix(
                        (np.ones(time1.size),
                         (np.zeros(time1.size), time1)),
                        shape = (1, N),
                        dtype = np.uint8).toarray()[0]
                    time2 = time_np[all_det_id == i2]
                    time2 = time2[time2 < N]
                    sgn2 = sparse.coo_matrix(
                        (np.ones(time2.size),
                         (np.zeros(time2.size), time2)),
                        shape = (1, N),
                        dtype = np.uint8).toarray()[0]
                    row = np.hstack((row, [d.size, d.size]))
                    col = np.hstack((col, [i1, i2]))
                    data = np.hstack(([data, [-1, 1]]))
                    d = np.hstack((d, max_shift_val(sgn1, sgn2, 25)[0]))
            A = sparse.coo_matrix((data, (row.astype(np.uint32), col.astype(np.uint32))))
            return A, -d


@nef.nef_class
class SubmoduleSolver:
    config: dict
    scanner: nef.PetCylindricalScanner

    def __call__(self, path: str) -> sparse.coo_matrix:
        nb_submodule = self.scanner.nb_submodule[0] * \
                       self.scanner.nb_submodule[1]
        nb_module = self.scanner.nb_module[0] * self.scanner.nb_module[1]
        nb_rsector = self.scanner.nb_rsector
        nb_rsector_half = nb_rsector // 2
        N = nb_module * nb_submodule * nb_rsector
        # data_mat = np.zeros(384 * 384, np.float32)
        # num_mat = np.zeros(384 * 384, np.uint32)
        data_mat = sparse.coo_matrix((N, N), dtype = np.float32)
        num_mat = sparse.coo_matrix((N, N), dtype = np.uint32)

        if 'num_data' not in self.config:
            raise ValueError('cannot find num_data field in config')
        else:
            pass

        for ind in nef.utils.tqdm(range(self.config['num_data'])):
            filename = path.replace('?', str(ind))
            time_true_np, rsector_id_np, module_id_np, submodule_id_np, crystal_id_np = dd.io.load(
                filename, [
                    '/time', '/rsector_id', '/module_id', '/submodule_id',
                    '/crystal_id'
                ])

            if 'sigma' in self.config:
                sigma = self.config['sigma']
                time_np = time_true_np + \
                          np.random.normal(0, sigma, time_true_np.size)
            else:
                time_np = time_true_np

            all_sub_np = submodule_id_np.astype(
                np.uint32) + nb_submodule * (module_id_np.astype(
                np.uint32) + nb_module * rsector_id_np.astype(np.uint32))
            all_id_np = crystal_id_np + self.scanner.nb_crystal[
                0] * self.scanner.nb_crystal[1] * all_sub_np
            all_det_id = module_id_np.astype(
                np.uint32) + nb_module * rsector_id_np.astype(np.uint32)

            if 'delay_super' in self.config:
                super_delay = self.config['delay_super']
                time_np += self.config['delay_super'][all_det_id] * 10000

            if 'estimated_delay_super' in self.config:
                time_np -= self.config['estimated_delay_super'][all_det_id] * 10000

            if 'delay_crystals' in self.config:
                time_np += self.config['delay_crystals'][all_id_np]

            if 'delay_submodule' in self.config:
                time_np += self.config['delay_submodule'][all_sub_np]

            if 'estimated_crystal_delay' in self.config:
                time_np -= self.config['estimated_crystal_delay'][all_id_np]

            if 'estimated_delay_submodule' in self.config:
                time_np -= self.config['estimated_delay_submodule'][all_sub_np]
            sort_ind = np.argsort(time_np)

            rsector_id_np = rsector_id_np[sort_ind]
            all_sub_np = all_sub_np[sort_ind]
            time_np = time_np[sort_ind]  # change time resolution to 100ps
            # condition_front = np.abs((rsector_id_np[1:] - rsector_id_np[:-1]) %
            #                          nb_rsector - nb_rsector_half) <= 1
            condition_3 = (rsector_id_np[1:] - rsector_id_np[:-1]) % 8 == 3
            condition_4 = (rsector_id_np[1:] - rsector_id_np[:-1]) % 8 == 4
            condition_5 = (rsector_id_np[1:] - rsector_id_np[:-1]) % 8 == 5

            condition_front = condition_3 + condition_4 + condition_5

            condition_window = time_np[1:] - time_np[:-1] < self.config[
                'time_window']
            lor_ind = np.where(
                np.logical_and(condition_front, condition_window))[0]
            # lor_ind_true = np.where(event_id[1:] == event_id[:-1])[0]
            fst_ind = lor_ind
            snd_ind = fst_ind + 1
            ind1 = all_sub_np[fst_ind]
            ind2 = all_sub_np[snd_ind]
            inv = ind1 > ind2

            row_all = ind1 * (1 - inv) + ind2 * inv
            col_all = ind1 * inv + ind2 * (1 - inv)
            dtime_all = (time_np[snd_ind] - time_np[fst_ind]) * 2 * (0.5 - inv)

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
        ans = np.nan_to_num(np.divide(data_mat, num_mat))[num_mat.nonzero()]
        return A, ans


@nef.nef_class
class CrystalSolver:
    config: dict
    scanner: nef.PetCylindricalScanner

    def __call__(self, path: str) -> sparse.coo_matrix:

        N = self.scanner.nb_all_crystal
        data_mat = sparse.coo_matrix((N, N), dtype = np.float32)
        num_mat = sparse.coo_matrix((N, N), dtype = np.uint32)
        nb_submodule = self.scanner.nb_submodule[0] * \
                       self.scanner.nb_submodule[1]
        nb_module = self.scanner.nb_module[0] * self.scanner.nb_module[1]
        nb_rsector = self.scanner.nb_rsector
        if 'num_data' not in self.config:
            raise ValueError('cannot find num_data field in config')
        else:
            pass

        for ind in nef.utils.tqdm(range(self.config['num_data'])):
            filename = path.replace('?', str(ind))
            time_true_np, rsector_id_np, module_id_np, submodule_id_np, crystal_id_np = dd.io.load(
                filename, [
                    '/time', '/rsector_id', '/module_id', '/submodule_id',
                    '/crystal_id'
                ])

            if 'sigma' in self.config:
                sigma = self.config['sigma']
                time_np = time_true_np + \
                          np.random.normal(0, sigma, time_true_np.size)
            else:
                time_np = time_true_np

            all_id_np = crystal_id_np + self.scanner.nb_crystal[
                0] * self.scanner.nb_crystal[1] * (
                                submodule_id_np.astype(np.uint32) +
                                self.scanner.nb_submodule[0] *
                                self.scanner.nb_submodule[1] *
                                (module_id_np.astype(np.uint32) +
                                 self.scanner.nb_module[0] * self.scanner.nb_module[1] *
                                 rsector_id_np.astype(np.uint32)))

            all_sub_np = submodule_id_np.astype(np.uint32) + (
                    self.scanner.nb_submodule[0] * self.scanner.nb_submodule[1] *
                    (module_id_np.astype(np.uint32) + self.scanner.nb_module[0] *
                     self.scanner.nb_module[1] * rsector_id_np.astype(np.uint32)))
            all_det_id = module_id_np.astype(
                np.uint32) + nb_module * rsector_id_np.astype(np.uint32)

            if 'delay_super' in self.config:
                super_delay = self.config['delay_super']
                time_np += self.config['delay_super'][all_det_id] * 10000

            if 'estimated_delay_super' in self.config:
                time_np -= self.config['estimated_delay_super'][all_det_id] * 10000

            if 'delay_crystals' in self.config:
                time_np += self.config['delay_crystals'][all_id_np]

            if 'delay_submodule' in self.config:
                time_np += self.config['delay_submodule'][all_sub_np]

            if 'estimated_delay_crystal' in self.config:
                time_np -= self.config['estimated_delay_crystal'][all_id_np]

            if 'estimated_delay_submodule' in self.config:
                time_np -= self.config['estimated_delay_submodule'][all_sub_np]

            sort_ind = np.argsort(time_np)
            rsector_id_np = rsector_id_np[sort_ind]
            all_id_np = all_id_np[sort_ind]
            time_np = time_np[sort_ind]

            condition_3 = (rsector_id_np[1:] - rsector_id_np[:-1]) % 8 == 3
            condition_4 = (rsector_id_np[1:] - rsector_id_np[:-1]) % 8 == 4
            condition_5 = (rsector_id_np[1:] - rsector_id_np[:-1]) % 8 == 5

            condition_345 = condition_3 + condition_4 + condition_5

            lor_ind = np.where(
                np.logical_and(
                    time_np[1:] - time_np[:-1] < self.config['time_window'],
                    condition_345))[0]
            # event_id = dd.io.load(filename, '/index')[sort_ind]
            # true_ind1 = event_id[lor_ind] == event_id[lor_ind + 1]
            # return lor_ind, true_ind1, time_true_np[
            #     sort_ind], rsector_id_np, event_id, time_np
            fst_ind = lor_ind
            snd_ind = fst_ind + 1
            fst_pos = nef.CylindricalIndexToCrystalPos(self.scanner)(
                all_id_np[fst_ind])
            snd_pos = nef.CylindricalIndexToCrystalPos(self.scanner)(
                all_id_np[snd_ind])
            x1 = fst_pos[:, 0]
            y1 = fst_pos[:, 1]
            z1 = fst_pos[:, 2]

            x2 = snd_pos[:, 0]
            y2 = snd_pos[:, 1]
            z2 = snd_pos[:, 2]
            L = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)
            a = (x2 - x1) ** 2 + (y2 - y1) ** 2
            b = 2 * x1 * (x2 - x1) + 2 * y1 * (y2 - y1)
            c = x1 ** 2 + y1 ** 2 - 100 ** 2
            k1 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / a / 2
            k2 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / a / 2
            is_not_nan_filter = np.logical_not(np.isnan(k1))
            fst_ind = fst_ind[is_not_nan_filter]
            snd_ind = snd_ind[is_not_nan_filter]
            a = a[is_not_nan_filter]
            b = b[is_not_nan_filter]
            c = c[is_not_nan_filter]
            ind1 = all_id_np[fst_ind]
            ind2 = all_id_np[snd_ind]
            inv = ind1 > ind2
            row_all = ind1 * (1 - inv) + ind2 * inv
            col_all = ind1 * inv + ind2 * (1 - inv)
            time1 = time_np[fst_ind]
            time2 = time_np[snd_ind]
            AE = (k1[is_not_nan_filter] +
                  k2[is_not_nan_filter]) / 2 * L[is_not_nan_filter]
            BE = L[is_not_nan_filter] - AE
            dtime_expect = (BE - AE) / 0.03
            dtime_all = (time2 - time1 - dtime_expect) * (0.5 - inv) * 2
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
        # new_col_all = np.hstack((data_mat.nonzero()[0], data_mat.nonzero()[1]))
        new_col_all = np.hstack(num_mat.nonzero())
        new_data_all = np.hstack(
            (np.ones(num_mat.nnz) * -1, np.ones(num_mat.nnz)))
        A = sparse.coo_matrix((new_data_all, (new_row_all, new_col_all)),
                              shape = (num_mat.nnz, N))
        # ans = data_mat.data / num_mat.data
        ans = np.nan_to_num(np.divide(data_mat, num_mat))[num_mat.nonzero()]
        return A, ans


@nef.nef_class
class SingleResolutionCalculator:
    config: dict
    scanner: nef.PetCylindricalScanner

    def __call__(self, path: str) -> sparse.coo_matrix:
        N = self.scanner.nb_all_crystal
        data_mat = sparse.coo_matrix((N, N), dtype = np.float32)
        num_mat = sparse.coo_matrix((N, N), dtype = np.uint32)
        nb_submodule = self.scanner.nb_submodule[0] * \
                       self.scanner.nb_submodule[1]
        nb_module = self.scanner.nb_module[0] * self.scanner.nb_module[1]
        nb_rsector = self.scanner.nb_rsector
        if 'num_data' not in self.config:
            raise ValueError('cannot find num_data field in config')
        else:
            pass

        for ind in nef.utils.tqdm(range(self.config['num_data'])):
            filename = path.replace('?', str(ind))
            time_true_np, rsector_id_np, module_id_np, submodule_id_np, crystal_id_np = dd.io.load(
                filename, [
                    '/time', '/rsector_id', '/module_id', '/submodule_id',
                    '/crystal_id'
                ])
            if 'sigma' in self.config:
                sigma = self.config['sigma']
                time_np = time_true_np + \
                          np.random.normal(0, sigma, time_true_np.size)
            else:
                time_np = time_true_np

            all_id_np = crystal_id_np + self.scanner.nb_crystal[
                0] * self.scanner.nb_crystal[1] * (
                                submodule_id_np.astype(np.uint32) +
                                self.scanner.nb_submodule[0] *
                                self.scanner.nb_submodule[1] *
                                (module_id_np.astype(np.uint32) +
                                 self.scanner.nb_module[0] * self.scanner.nb_module[1] *
                                 rsector_id_np.astype(np.uint32)))

            all_sub_np = submodule_id_np.astype(np.uint32) + (
                    self.scanner.nb_submodule[0] * self.scanner.nb_submodule[1] *
                    (module_id_np.astype(np.uint32) + self.scanner.nb_module[0] *
                     self.scanner.nb_module[1] * rsector_id_np.astype(np.uint32)))
            all_det_id = module_id_np.astype(
                np.uint32) + nb_module * rsector_id_np.astype(np.uint32)
            if 'delay_super' in self.config:
                super_delay = self.config['delay_super']
                time_np += self.config['delay_super'][all_det_id] * 10000

            if 'estimated_delay_super' in self.config:
                time_np -= self.config['estimated_delay_super'][all_det_id] * 10000

            if 'delay_crystals' in self.config:
                time_np += self.config['delay_crystals'][all_id_np]

            if 'delay_submodule' in self.config:
                time_np += self.config['delay_submodule'][all_sub_np]

            if 'estimated_delay_crystal' in self.config:
                time_np -= self.config['estimated_delay_crystal'][all_id_np]

            if 'estimated_delay_submodule' in self.config:
                time_np -= self.config['estimated_delay_submodule'][all_sub_np]

            return time_np - time_true_np


@nef.nef_class
class TOFResolutionCalculator:
    config: dict
    scanner: nef.PetCylindricalScanner

    def __call__(self, path: str) -> sparse.coo_matrix:
        N = self.scanner.nb_all_crystal
        data_mat = sparse.coo_matrix((N, N), dtype = np.float32)
        num_mat = sparse.coo_matrix((N, N), dtype = np.uint32)
        nb_submodule = self.scanner.nb_submodule[0] * \
                       self.scanner.nb_submodule[1]
        nb_module = self.scanner.nb_module[0] * self.scanner.nb_module[1]
        nb_rsector = self.scanner.nb_rsector
        if 'num_data' not in self.config:
            raise ValueError('cannot find num_data field in config')
        else:
            pass

        for ind in nef.utils.tqdm(range(self.config['num_data'])):
            filename = path.replace('?', str(ind))
            time_true_np, rsector_id_np, module_id_np, submodule_id_np, crystal_id_np = dd.io.load(
                filename, [
                    '/time', '/rsector_id', '/module_id', '/submodule_id',
                    '/crystal_id'
                ])
            if 'sigma' in self.config:
                sigma = self.config['sigma']
                time_np = time_true_np + \
                          np.random.normal(0, sigma, time_true_np.size)
            else:
                time_np = time_true_np

            all_id_np = crystal_id_np + self.scanner.nb_crystal[
                0] * self.scanner.nb_crystal[1] * (
                                submodule_id_np.astype(np.uint32) +
                                self.scanner.nb_submodule[0] *
                                self.scanner.nb_submodule[1] *
                                (module_id_np.astype(np.uint32) +
                                 self.scanner.nb_module[0] * self.scanner.nb_module[1] *
                                 rsector_id_np.astype(np.uint32)))

            all_sub_np = submodule_id_np.astype(np.uint32) + (
                    self.scanner.nb_submodule[0] * self.scanner.nb_submodule[1] *
                    (module_id_np.astype(np.uint32) + self.scanner.nb_module[0] *
                     self.scanner.nb_module[1] * rsector_id_np.astype(np.uint32)))
            all_det_id = module_id_np.astype(
                np.uint32) + nb_module * rsector_id_np.astype(np.uint32)
            if 'delay_super' in self.config:
                super_delay = self.config['delay_super']
                time_np += self.config['delay_super'][all_det_id] * 10000

            if 'estimated_delay_super' in self.config:
                time_np -= self.config['estimated_delay_super'][all_det_id] * 10000

            if 'delay_crystals' in self.config:
                time_np += self.config['delay_crystals'][all_id_np]

            if 'delay_submodule' in self.config:
                time_np += self.config['delay_submodule'][all_sub_np]

            if 'estimated_delay_crystal' in self.config:
                time_np -= self.config['estimated_delay_crystal'][all_id_np]

            if 'estimated_delay_submodule' in self.config:
                time_np -= self.config['estimated_delay_submodule'][all_sub_np]

            event_id = dd.io.load(filename, '/index')
            fst_ind = np.where(event_id[:-1] == event_id[1:])[0]
            snd_ind = fst_ind + 1

            dtime = time_np[snd_ind] - time_np[fst_ind]
            dtime_true = time_true_np[snd_ind] - time_true_np[fst_ind]

            return dtime - dtime_true
