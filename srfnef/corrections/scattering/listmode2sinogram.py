import math

import numpy as np
from numba import jit, cuda
import srfnef as nef



def change_id(id1,id2):
    index = np.where(id1<id2)[0]
    maxs = id2[index]
    id2[index] = id1[index]
    id1[index] = maxs
    return id1.reshape(-1,1),id2.reshape(-1,1)


@jit(nopython = True)
def cal_sinogram(list_mode_data, total_num_crystal):
    weight = np.zeros((int(total_num_crystal * (total_num_crystal - 1) / 2), 1))
    for i in range(int(list_mode_data.shape[0])):
        pos = int((list_mode_data[i, 0] - 1) * list_mode_data[i, 0] / 2 + list_mode_data[i, 1])
        weight[pos] = weight[pos] + list_mode_data[i, 2]
    return weight


@jit(nopython = True)
def get_all_lors_id(total_num_crystal):
    lors = np.zeros((int(total_num_crystal * (total_num_crystal - 1) / 2), 2), dtype = np.int32)
    num = 0
    for i in range(total_num_crystal):
        for j in range(i):
            lors[num, 0] = i
            lors[num, 1] = j
            num = num + 1
    return lors


def get_crystal_xy(grid_block, num_block, crystal_id, block_id, r_inner, size_block):
    size_crystal = size_block / grid_block
    angle_block = math.pi * 2 / num_block * block_id
    center = np.hstack((((r_inner) * np.cos(angle_block)).reshape(block_id.size, 1),
                        ((r_inner) * np.sin(angle_block)).reshape(block_id.size, 1)))
    vector_tang = np.hstack(
        (np.cos(angle_block + math.pi / 2 * np.ones_like(angle_block)).reshape(block_id.size, 1),
         np.sin(angle_block + math.pi / 2 * np.ones_like(angle_block)).reshape(block_id.size, 1)))
    dist_from_center = size_crystal[1] * (
            crystal_id - (grid_block[1] - 1) / 2 * np.ones_like(crystal_id))
    return center + (np.hstack((dist_from_center.reshape(dist_from_center.size, 1),
                                dist_from_center.reshape(dist_from_center.size, 1)))
                     * vector_tang)


def get_crystal_z(ring_id, grid_block, size_block, nb_rings):
    size_crystal = size_block / grid_block
    dist_from_center = (ring_id - (grid_block[2] * nb_rings - 1) / 2 * np.ones_like(ring_id)) * \
                       size_crystal[2]
    return dist_from_center


def get_center(scanner, crystal_id_whole_scanner):
    ring_id = np.floor(
        crystal_id_whole_scanner / scanner.nb_blocks_per_ring / scanner.blocks.shape[1])
    crystal_per_ring_id = crystal_id_whole_scanner - ring_id * scanner.nb_blocks_per_ring * \
                          scanner.blocks.shape[1]
    block_id = crystal_per_ring_id // scanner.blocks.shape[1]
    crystal_id = crystal_per_ring_id % scanner.blocks.shape[1]
    center_xy = get_crystal_xy(np.array(scanner.blocks.shape), scanner.nb_blocks_per_ring,
                               crystal_id, block_id, scanner.inner_radius,
                               np.array(scanner.blocks.size))
    center_z = get_crystal_z(ring_id, np.array(scanner.blocks.shape), np.array(scanner.blocks.size),
                             scanner.nb_rings)
    return np.hstack((center_xy, center_z.reshape(center_z.size, 1)))


#@jit
def lm2sino(listmode, scanner):
    if isinstance(scanner, nef.PetEcatScanner):
        id1 = nef.EcatCrystalPosToIndex(scanner)(listmode.lors.data[:,:3])
        id2 = nef.EcatCrystalPosToIndex(scanner)(listmode.lors.data[:,3:6])
    elif isinstance(scanner, nef.PetCylindricalScanner):
        id1 = nef.CylindricalCrystalPosToIndex(scanner)(listmode.lors.data[:,:3])
        id2 = nef.CylindricalCrystalPosToIndex(scanner)(listmode.lors.data[:,3:6])
    id1_,id2_ = change_id(id1,id2)
    listmode_data = np.hstack((np.hstack((id1_,id2_)), listmode.data.reshape(-1, 1)))
    return cal_sinogram(listmode_data, scanner.nb_crystals)


def sino2lm(scanner, sinogram, lors):
    index = np.where((sinogram > 0)&(sinogram < 1e8))[0]
    all_position = np.zeros((len(index), 6))
    # all_position[:, :3] = get_center(scanner, lors[index, 0])
    # all_position[:, 3:6] = get_center(scanner, lors[index, 1])
    if isinstance(scanner, nef.PetEcatScanner):
        all_position[:, :3] = nef.EcatIndexToCrystalPos(scanner)(lors[index, 0])
        all_position[:, 3:6] = nef.EcatIndexToCrystalPos(scanner)(lors[index, 1])
    elif isinstance(scanner, nef.PetCylindricalScanner):
        all_position[:, :3] = nef.CylindricalIndexToCrystalPos(scanner)(lors[index, 0])
        all_position[:, 3:6] = nef.CylindricalIndexToCrystalPos(scanner)(lors[index, 1])   
    return np.hstack((all_position, sinogram[index].reshape(-1, 1)))


__all__ = []

DEFAULT_GROUP_NAME = 'listmode_data'

DEFAULT_COLUMNS = ['fst', 'snd', 'weight', 'tof']
