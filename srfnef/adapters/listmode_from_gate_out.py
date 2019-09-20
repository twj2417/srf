# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: listmode_from_gate_out.py
@date: 5/21/2019
@desc:
'''
import numpy as np
from srfnef.utils import tqdm
from srfnef import Lors, Listmode, PetEcatScanner,  save
import srfnef as nef


# def listmode_trans(path, scanner, nb_sub):
#     full_data = None
#     path_file = path + 'sub.?/result_with_scatter.npy'
#     for i in tqdm(range(nb_sub)):
#         path_ = path_file.replace('?', str(i))
#         data = np.load(path_)[:, :6]
#         if full_data is None:
#             full_data = np.array(data)
#         else:
#             full_data = np.vstack((full_data, data))
#     lors = Lors(full_data)
#     listmode = Listmode(np.ones((lors.length,), dtype = np.float32), lors)
#     row, col = ListmodeToId()(listmode, scanner)

#     lors = IdToListmode()(row, col, scanner)
#     out = Listmode(np.ones((lors.shape[0],), dtype = np.float32), lors)
#     save(out, path + 'listmode_scatter.hdf5')
#     return out


def listmode_from_gate_out(path, scanner, nb_sub):
    full_data = None
    for i in tqdm(range(nb_sub)):
        path_ = path.replace('?', str(i))
        data = np.load(path_)[:, :7]
        if full_data is None:
            full_data = np.array(data)
        else:
            full_data = np.vstack((full_data, data))
    if isinstance(scanner, nef.PetEcatScanner):
        lors = Lors(full_data)
        listmode = Listmode(np.ones((lors.length,), dtype = np.float32), lors)
        row = nef.EcatCrystalPosToIndex(scanner)(lors.data[:, :3])
        col = nef.EcatCrystalPosToIndex(scanner)(lors.data[:, 3:])
        lors_data1 = nef.EcatIndexToCrystalPos(scanner)(row)
        lors_data2 = nef.EcatIndexToCrystalPos(scanner)(col)
        lors = np.hstack((lors_data1, lors_data2))
        return Listmode(np.ones((lors.shape[0],), dtype = np.float32), lors)
    elif isinstance(scanner, nef.PetCylindricalScanner):
        lors = Lors(full_data)
        listmode = Listmode(np.ones((lors.length,), dtype = np.float32), lors)
        ind1 = nef.CylindricalCrystalPosToIndex(scanner)(lors.data[:, :3])
        ind2 = nef.CylindricalCrystalPosToIndex(scanner)(lors.data[:, 3:])
        lors_data1 = nef.CylindricalIndexToCrystalPos(scanner)(ind1)
        lors_data2 = nef.CylindricalIndexToCrystalPos(scanner)(ind2)
        lors_data = np.hstack((lors_data1, lors_data2))
        return listmode.update(lors = nef.Lors(lors_data))
    else:
        raise NotImplementedError


#
# def listmode_from_gate_out(path, scanner, nb_sub):
#     full_data = None
#     for i in tqdm(range(nb_sub)):
#         path_ = path.replace('?', str(i))
#         data = np.load(path_)[:, :7]
#         if full_data is None:
#             full_data = np.array(data)
#         else:
#             full_data = np.vstack((full_data, data))
#     lors = Lors(full_data)
#     listmode = Listmode(np.ones((lors.length,), dtype = np.float32), lors)
#     row, col = ListmodeToId()(listmode, scanner)
#     lors = IdToListmode()(row, col, scanner)
#     return Listmode(np.ones((lors.shape[0],), dtype = np.float32), lors)


def listmode_tof_from_gate_out(path, scanner, nb_sub):
    full_data = None
    for i in tqdm(range(nb_sub)):
        path_ = path.replace('?', str(i))
        data = np.load(path_)[:, :7]
        if full_data is None:
            full_data = np.array(data)
        else:
            full_data = np.vstack((full_data, data))
    lors = Lors(full_data)
    listmode = Listmode(np.ones((lors.length,), dtype = np.float32), lors)
    row, col = ListmodeToId()(listmode, scanner)
    lors = IdToListmode()(row, col, scanner)
    print(lors.data.shape)
    print(full_data.shape)
    lors_tof = lors.update(data = np.append(lors.data, full_data[:, -1].reshape(-1, 1), axis = 1))
    return Listmode(np.ones((lors.shape[0],), dtype = np.float32), lors_tof)


def listmode_from_gate_out_multi_bed(path, scanner, nb_sub):
    full_data = {}
    for i in tqdm(range(nb_sub)):
        path_ = path.replace('?', str(i))
        data = np.load(path_)[:, :7]
        bed_id = np.load(path_)[:, -1]
        for i_bed in set(bed_id):
            if i_bed not in full_data:
                full_data[i_bed] = np.array(data[bed_id == i_bed, :])
            else:
                full_data[i_bed] = np.vstack((full_data[i_bed], data[bed_id == i_bed, :]))

    listmode_out = {}
    for key, values in full_data.items():
        lors = Lors(values)
        listmode = Listmode(np.ones((lors.length,), dtype = np.float32), lors)
        row, col = ListmodeToId()(listmode, scanner)
        lors = IdToListmode()(row, col, scanner)
        listmode_out[key] = Listmode(np.ones((lors.shape[0],), dtype = np.float32), lors)
    return listmode_out
    #     #
    #     # if full_data is None:
    #     #     full_data = np.array(data)
    #     # else:
    #     #     full_data = np.vstack((full_data, data))
    # lors = Lors(full_data)
    # listmode = Listmode(np.ones((lors.length,), dtype = np.float32), lors)
    # row, col = ListmodeToId()(listmode, scanner)
    # lors = IdToListmode()(row, col, scanner)
    # return Listmode(np.ones((lors.shape[0],), dtype = np.float32), lors)
