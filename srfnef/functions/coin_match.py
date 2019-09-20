# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: coin_match.py
@date: 9/3/2019
@desc:
'''
import srfnef as nef
import numpy as np
from srfnef import nef_class, PetCylindricalScanner
from srfnef.adapters.gate_single_loader import GateSingleLoader
from scipy.sparse import coo_matrix


@nef_class
class CoinMatcher:
    time_window: float
    scanner: PetCylindricalScanner
    loader: GateSingleLoader

    def __call__(self, num: int, filename: str, random_interval: int = None) -> object:

        fst_pos_all = np.zeros((0, 3))
        snd_pos_all = np.zeros((0, 3))
        tof_all = np.array([])
        val_all = np.array([])
        for ind in nef.utils.tqdm(range(num)):
            filename_ = filename.replace('?', str(ind))
            time_, crystal_id = self.loader(filename_)
            fst_ind = np.where(time_[1:] - time_[:-1] < self.time_window)[0]
            snd_ind = fst_ind + 1
            fst_crystal_id = crystal_id[fst_ind]
            snd_crystal_id = crystal_id[snd_ind]
            tof_dist = (time_[snd_ind] - time_[fst_ind]) * 0.3
            tof_all = np.hstack((tof_all, tof_dist))
            fst_pos = nef.CylindricalIndexToCrystalPos(self.scanner)(fst_crystal_id)
            snd_pos = nef.CylindricalIndexToCrystalPos(self.scanner)(snd_crystal_id)
            fst_pos_all = np.vstack((fst_pos_all, fst_pos))
            snd_pos_all = np.vstack((snd_pos_all, snd_pos))

            if random_interval is None:
                val_all = np.hstack((val_all, np.ones(fst_ind.size))).astype(np.float32)
            elif random_interval == 0:
                N = self.scanner.nb_all_crystal
                single_rate = np.zeros(N)
                for ind in range(N):
                    single_rate[ind] = np.sum(crystal_id == ind)
                val_all = 1 - single_rate[fst_crystal_id] * single_rate[
                    snd_crystal_id] * 2 * self.time_window / (time_[-1] - time_[0])
            else:
                if ind % random_interval == 0:
                    N = self.scanner.nb_all_crystal
                    mat_coin = coo_matrix((N, N), dtype = np.float32)
                    mat_rand = coo_matrix((N, N), dtype = np.float32)
                    for ind in nef.utils.tqdm(range(0, num, random_interval)):
                        filename_ = filename.replace('?', str(ind))
                        time_, crystal_id = self.loader(filename_)
                        fst_ind = np.where(time_[1:] - time_[:-1] < self.time_window)[0]
                        snd_ind = fst_ind + 1
                        mat_coin += coo_matrix((np.ones(fst_ind.size),
                                                (crystal_id[fst_ind], crystal_id[snd_ind])), (N, N),
                                               dtype = np.float32).tocsr()
                        panel_id = crystal_id // self.scanner.nb_crystal_per_rsector
                        time_rand = time_ + panel_id * 5 * self.time_window
                        ind_sort = np.argsort(time_rand)
                        time_rand = time_rand[ind_sort]
                        crystal_id_rand = crystal_id[ind_sort]
                        fst_ind = np.where(time_rand[1:] - time_rand[:-1] < self.time_window)[0]
                        snd_ind = fst_ind + 1
                        fst_crystal_id_rand = crystal_id_rand[fst_ind]
                        snd_crystal_id_rand = crystal_id_rand[snd_ind]
                        mat_rand += coo_matrix(
                            (np.ones(fst_crystal_id_rand.size),
                             (fst_crystal_id_rand, snd_crystal_id_rand)),
                            (N, N), dtype = np.float32).tocsr()
                vals_ = np.zeros(fst_ind.size)
                for ind in range(vals_.size):
                    vals_[ind] = np.nan_to_num(1 - mat_rand[fst_crystal_id[ind], snd_crystal_id[
                        ind]] / mat_coin[fst_crystal_id[ind], snd_crystal_id[ind]])
                val_all = np.hstack((val_all, vals_.astype(np.float32)))
                mat_coin.sum_duplicates()
                mat_rand.sum_duplicates()
        lors_data = np.hstack((fst_pos_all, snd_pos_all, tof_all.reshape(-1, 1))).astype(np.float32)
        return nef.Listmode(val_all.ravel(), nef.Lors(lors_data))
