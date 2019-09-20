import numpy as np
import srfnef
from .listmode2sinogram import get_center
import srfnef as nef

def pre_all_scatter_position(emission_image):
    size_voxel = np.array(emission_image.unit_size)
    step = np.floor(50 / size_voxel)
    time = np.floor(emission_image.shape / step)
    position = np.zeros(
        (int(time[0] * time[1] * time[2]), 3), dtype = np.float32)
    n = 0
    for x in range(int(time[0])):
        for y in range(int(time[1])):
            for z in range(int(time[2])):
                position[n, :] = get_coordinate(emission_image, step, x, y, z)
                n = n + 1
    return position


def pre_sumup_of_emission(emission_image, p1, p2):
    lors = get_lors(p1, p2)
    projector = srfnef.functions.Project('tf-eager')
    value = projector(emission_image, lors).data
    len_lors = np.power(lors.data[:, 3] - lors.data[:, 0], 2) + np.power(
        lors.data[:, 4] - lors.data[:, 1], 2) + np.power(lors.data[:, 5] - lors.data[:, 2], 2)
    return value * len_lors


def pre_atten(transmission_image, p1, p2):
    lors = get_lors(p1, p2)
    u_map_projector = srfnef.corrections.attenuation.UmapProject()
    value = u_map_projector(transmission_image, lors).data
    return value


def pre_lors_atten(transmission_image, p1, p2):
    lors = np.hstack((np.hstack((p1, p2)), np.ones(
        (p1.shape[0], 1), dtype = np.float32)))
    u_map_projector = srfnef.corrections.attenuation.UmapProject()
    value = u_map_projector(transmission_image * 1.3,
                            srfnef.data.Lors(lors)).data
    return value


def get_lors(P1, P2):
    lors = np.ones((P1.shape[0] * P2.shape[0], 7), dtype = np.float32)
    lors[:, 0] = np.tile(P1[:, 0].reshape(-1, 1), P2.shape[0]).flatten()
    lors[:, 1] = np.tile(P1[:, 1].reshape(-1, 1), P2.shape[0]).flatten()
    lors[:, 2] = np.tile(P1[:, 2].reshape(-1, 1), P2.shape[0]).flatten()
    lors[:, 3] = np.tile(P2[:, 0], P1.shape[0])
    lors[:, 4] = np.tile(P2[:, 1], P1.shape[0])
    lors[:, 5] = np.tile(P2[:, 2], P1.shape[0])
    return srfnef.data.Lors(lors)


def get_coordinate(emission_image, step, x, y, z):
    size_pixel = np.array(emission_image.unit_size)
    emission_image_size = np.array(emission_image.size)
    return np.array([step[0] * (x + 1 / 2) * size_pixel[0], step[1] * (y + 1 / 2) * size_pixel[1],
                     step[2] * (z + 1 / 2) * size_pixel[2]]) - emission_image_size / 2


def get_voxel_volume(image):
    size_pixel = np.array(image.unit_size)
    return size_pixel[0] * size_pixel[1] * size_pixel[2]


def get_all_crystal_position(scanner):
    crystal_id = np.arange(scanner.nb_crystals)
    if isinstance(scanner, nef.PetEcatScanner):
        all_position = nef.EcatIndexToCrystalPos(scanner)(crystal_id)
    elif isinstance(scanner, nef.PetCylindricalScanner):
        all_position = nef.CylindricalIndexToCrystalPos(scanner)(crystal_id)
    return all_position.astype(np.float32)
