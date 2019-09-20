import math

import numpy as np
from numba import jit, cuda

from .kn_formula import get_scatter_cos_theta, fkn, fkn_new

import tensorflow as tf



@jit
def scatter_fraction(emission_image, u_map, index, lors, nb_blocks_per_ring,nb_crystals_per_ring,grid_block,size_block, 
                            low_energy, high_energy,resolution,scatter_position,crystal_position,sumup_emission,atten):
    scatter = np.zeros((index.size, 1), dtype = np.float32)
    scale = np.zeros((index.size, 1), dtype = np.float32)
    lors_part = lors[index, :]
    loop_all_lors[(512, 512), (16, 16)](nb_crystals_per_ring,
                                        nb_blocks_per_ring,
                                        grid_block,
                                        scatter_position, crystal_position, low_energy, high_energy,
                                        math.sqrt(511) * resolution,
                                        atten, sumup_emission, scatter_position.shape[0],
                                        u_map.data, np.array(u_map.size, dtype = np.float32),
                                        np.array(u_map.data.shape, dtype = np.int32),
                                        lors_part, scatter, scale)
    efficiency_without_scatter = eff_without_scatter(low_energy, high_energy,
                                                     math.sqrt(511) * resolution) * 5 / (
                                         1022 * math.pi) ** 0.5 / resolution
    scatter = scatter * efficiency_without_scatter * (
            size_block[1] * size_block[2] / grid_block[1] /
            grid_block[2]) ** 2 * 5 / (
                      1022 * math.pi) ** 0.5 / resolution / 4 / math.pi
    scale = scale * 4 * math.pi / efficiency_without_scatter ** 2 / (
            size_block[1] * size_block[2] / grid_block[1] /
            grid_block[2]) ** 2
    return scatter, scale


@cuda.jit
def loop_all_lors(nb_crystals_per_ring, nb_blocks_per_ring, grid_block, scatter_position,
                  crystal_position, low_energy, high_energy,
                  resolution, atten, sumup_emission, num_scatter, u_map, size, shape, lors,
                  scatter_ab, scale_ab):

#循环所有lor对，对没对lor计算散射概率
    c, j = cuda.grid(2)
    i = 512 * 16 * c + j
    if i < scatter_ab.shape[0]:
        a = int(lors[i, 0])
        b = int(lors[i, 1])
        scatter_ab[i, 0] = loop_all_s(scatter_position, crystal_position[a, :],
                                      crystal_position[b, :], low_energy, high_energy, resolution,
                                      sumup_emission[a * num_scatter:(a + 1) * num_scatter],
                                      sumup_emission[b * num_scatter:(b + 1) * num_scatter],
                                      atten[a * num_scatter:(a + 1) * num_scatter],
                                      atten[b * num_scatter:(b + 1) * num_scatter],
                                      nb_crystals_per_ring, nb_blocks_per_ring, grid_block, u_map,
                                      size, shape)
        scale_ab[i, 0] = get_scale(crystal_position[a, :], crystal_position[b, :],
                                   nb_crystals_per_ring, nb_blocks_per_ring, grid_block)


@cuda.jit(device = True)
def get_scale(A, B, nb_crystals_per_ring, nb_blocks_per_ring, grid_block):
    area = project_area(nb_crystals_per_ring, nb_blocks_per_ring, grid_block, A, B)
    return (distance_a2b(A[0], A[1], A[2], B[0], B[1], B[2]) / area) ** 2


@cuda.jit(device = True)
def loop_all_s(scatter_position, A, B, low_energy, high_energy, resolution, sumup_emission_s2a,
               sumup_emission_s2b, atten_s2a, atten_s2b,
               nb_crystals_per_ring, nb_blocks_per_ring, grid_block, u_map, size, shape):
#对于一对特定的lor，遍历所有散射点，计算各个散射点对该lor的贡献
    scatter_ab = 0
    for i in range(int(scatter_position.shape[0])):
        S = scatter_position[i, :]
        cos_theta = math.cos(get_scatter_cos_theta(A, S, B))
        scattered_energy = 511 / (2 - cos_theta)
        Ia = atten_s2a[i] * math.exp(math.log(atten_s2b[i]) / scattered_energy * 511) * \
             sumup_emission_s2a[i]
        Ib = math.exp(math.log(atten_s2a[i]) / scattered_energy * 511) * atten_s2b[i] * \
             sumup_emission_s2b[i]
        scatter_ab += (project_area(nb_crystals_per_ring, nb_blocks_per_ring, grid_block, S,
                                    A) * eff_with_scatter(low_energy, high_energy, scattered_energy,
                                                          resolution) *
                       project_area(nb_crystals_per_ring, nb_blocks_per_ring, grid_block, S,
                                    B) * fkn(A, S, B, u_map, size, shape) * (Ia + Ib) /
                       (distance_a2b(S[0], S[1], S[2], A[0], A[1], A[2])) ** 2 / (
                           distance_a2b(S[0], S[1], S[2], B[0], B[1], B[2])) ** 2)
    return scatter_ab


@jit(nopython = True)
def eff_without_scatter(low_energy_window, high_energy_window, energy_resolution):
#calulate detection efficiency according to lors energy with no scatter
    eff = 0
    for i in range(low_energy_window, high_energy_window, 5):
        eff += math.exp(-(float(i) - 511) ** 2 / 2 / energy_resolution ** 2)
    return eff


@cuda.jit(device = True)
def eff_with_scatter(low_energy_window, high_energy_window, scattered_energy, energy_resolution):
#calulate detection efficiency according to lors energy with scatter
    eff = 0
    for i in range(low_energy_window, high_energy_window, 5):
        eff += math.exp(-(float(i) - scattered_energy) ** 2 / 2 / energy_resolution ** 2)
    return eff


@cuda.jit(device = True)
def project_area(nb_crystals_per_ring, nb_blocks_per_ring, grid_block, pa, pb):
#calculate LOR ab projection area on pb,which is crystal area*cos(theta)

    theta_normal = get_block_theta(nb_crystals_per_ring, nb_blocks_per_ring, grid_block, pb)
    theta = (((pb[0] - pa[0]) * math.cos(theta_normal) + (pb[1] - pa[1]) * math.sin(theta_normal))
             / distance_a2b(pa[0], pa[1], pa[2], pb[0], pb[1], pb[2]))
    return theta


@cuda.jit(device = True)
def get_block_theta(nb_crystals_per_ring, nb_blocks_per_ring, grid_block, p):
    event_norm = distance_a2b(p[0], p[1], p[2], 0, 0, 0)
    before_theta = p[0] / event_norm
    theta_event = math.acos(before_theta) / math.pi * 180
    if p[1] < 0:
        theta_event = 360 - theta_event
    fixed_theta = (theta_event + 180 / nb_crystals_per_ring * grid_block[1]) % 360
    id_block = math.floor(fixed_theta / 360 * nb_blocks_per_ring)
    return id_block / nb_blocks_per_ring * 2 * math.pi


@cuda.jit(device = True)
def distance_a2b(x1, y1, z1, x2, y2, z2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2) ** 0.5
