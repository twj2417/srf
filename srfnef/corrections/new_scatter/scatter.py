import sys

import numpy as np
import srfnef as nef
from srfnef import nef_class
from srfnef.data import Image, Umap, Listmode, Lors
from srfnef.geometry import PetEcatScanner
import tensorflow as tf
import math
from numba import jit
from srfnef.config import TF_USER_OP_PATH

scatter_module = tf.load_op_library(TF_USER_OP_PATH + '/tf_scatter_module.so')
scatter_op = scatter_module.scatter_over_lors
scale_op = scatter_module.scale_over_lors


@nef_class
class ScatterCorrect:
    low_energy: float
    high_energy: float
    resolution: float
    scatter_fraction: float

    def __call__(self, emission_image: Image, u_map: Umap, scanner: PetEcatScanner,
                 listmode: Listmode) -> Listmode:
        nef.utils.declare_eager_execution()
        lors2 = nef.ScannerToCrystal()(nef.scanner_mct)

        # sampled scattering points
        smp = np.floor(np.array([50,50,50])/emission_image.unit_size)
        # ix = np.arange(0, emission_image.shape[0], smp[0])
        # iy = np.arange(0, emission_image.shape[1], smp[1])
        # iz = np.arange(0, emission_image.shape[2], smp[2])
        # x1, y1, z1 = np.meshgrid(ix, iy, iz, indexing = 'ij')
        # x1 = ((x1 + 0.5) *
        #       emission_image.unit_size[0] - emission_image.size[0] / 2).ravel()
        # y1 = ((y1 + 0.5) *
        #       emission_image.unit_size[1] - emission_image.size[1] / 2).ravel()
        # z1 = ((z1 + 0.5) *
        #       emission_image.unit_size[2] - emission_image.size[2] / 2).ravel()
    # crystal generate lors between crystal and scattering points
        # lors_data = np.zeros((x1.shape[0] * lors2.shape[0], 6), np.float32)
        # lors_data[:, 0] = np.kron(x1, [1] * lors2.shape[0])
        # lors_data[:, 1] = np.kron(y1, [1] * lors2.shape[0])
        # lors_data[:, 2] = np.kron(z1, [1] * lors2.shape[0])
        # lors_data[:, 3] = np.kron(lors2[:, 0], [[1]] * x1.shape[0]).ravel()
        # lors_data[:, 4] = np.kron(lors2[:, 1], [[1]] * x1.shape[0]).ravel()
        # lors_data[:, 5] = np.kron(lors2[:, 2], [[1]] * x1.shape[0]).ravel()
        # lors_smp = nef.Lors(lors_data)
    # pre attenuation
        # atten_proj = nef.UmapProject(u_map,lors_smp)
        # emiss_proj = nef.Project('tf-eager')(emission_image, lors_smp)
        # umap_project_tf = tf.constant(atten_proj.data)
        # image_project_tf = tf.constant(emiss_proj.data)
        # lors_tf = tf.transpose(tf.constant(listmode.lors.data))
        # u_map_tf = tf.transpose(tf.constant(u_map.data))

        ind1 = nef.CrystalToId()(listmode.lors.data[:, :3], nef.scanner_mct)
        ind2 = nef.CrystalToId()(listmode.lors.data[:, 3:], nef.scanner_mct)
        pv_tf = scatter_op(lors = lors_tf, ind1 = ind1, ind2 = ind2,
                           umap_project = umap_project_tf,
                           image_project = image_project_tf,
                           umap = u_map_tf,
                           grid = emission_image.shape,
                           center = emission_image.center,
                           size = emission_image.size,
                           smp = smp,
                           low_eng = self.low_energy,
                           high_eng = self.high_energy,
                           res_eng = self.resolution,
                           angle_per_block = np.pi * 2 / scanner.nb_blocks_per_ring,
                           crystal_area = scanner.blocks.unit_size[1] *
                                          scanner.blocks.unit_size[2])
        fraction = pv_tf.numpy()
        epsilon_ab = eff_without_scatter(self.low_energy, self.high_energy,
                                         math.sqrt(511) * self.resolution)
        sv_tf = scale_op(lors = lors_tf,
                         angle_per_block = np.pi * 2 / scanner.nb_blocks_per_ring,
                         crystal_area = scanner.blocks.unit_size[1] *
                                        scanner.blocks.unit_size[2],
                         epsilon_ab = 1.0)

        scale = sv_tf.numpy()
        return fraction, scale
        sum_fraction = np.sum(fraction)
        sum_listmode = np.sum(listmode.data)
        corrected_listmode_data = (listmode.data - fraction / sum_fraction * sum_listmode *
                                   self.scatter_fraction) * scale
        # atten_proj_full = nef.Project('tf-eager')(u_map, listmode.lors)

        # corrected_listmode_data = (listmode.data - fraction) * scale * atten_proj_full.data
        corrected_listmode = listmode.update(data = corrected_listmode_data)

        return nef.AttenuationCorrect(u_map)(corrected_listmode)


@jit(nopython = True)
def eff_without_scatter(low_energy_window, high_energy_window, energy_resolution):
    """
    calulate detection efficiency according to lors energy with no scatter
    """
    eff = 0
    for i in range(low_energy_window, high_energy_window, 5):
        eff += math.exp(-(float(i) - 511) ** 2 / 2 / energy_resolution ** 2) \
               / math.sqrt(2 * math.pi * energy_resolution ** 2) * 5
    return eff


def pre_lors_atten(transmission_image, p1, p2):
    lors = np.hstack((np.hstack((p1, p2)), np.ones(
        (p1.shape[0], 1), dtype = np.float32)))
    u_map_projector = nef.corrections.attenuation.UmapProject()
    value = u_map_projector(transmission_image * 1.3,
                            nef.data.Lors(lors)).data
    return value
