import numpy as np
import deepdish as dd
import srfnef as nef
from scipy import sparse


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

            # start from here
            sort_ind = np.argsort(time_np)
            time_np = (time_np[sort_ind] / 10000).astype(np.int32)  # change to ns
            all_sub_np = all_sub_np[sort_ind]
            all_det_id = all_det_id[sort_ind]
            N = 10 ** 7

            # A = np.zeros((32, 32))
            row, col, data = np.array([]), np.array([]), np.array([])
            d = np.array([])
            for i1 in nef.utils.tqdm(range(32)):
                for i2 in range(i1, 32):
                    if (i1 // 4 - i2 // 4) % 8 not in [3, 4, 5]:  # fine opposite panels
                        continue
                    time1 = time_np[all_det_id == i1]
                    time1 = time1[time1 < N]  # concern first 0.01s data
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
