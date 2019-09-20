# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: attenuation_correct.py
@date: 5/7/2019
@desc:
'''
import numpy as np
from srfnef import nef_class
from srfnef.functions import Project
from srfnef.data import Listmode, Umap


@nef_class
class AttenuationCorrect:
    u_map: Umap

    def __call__(self, listmode: Listmode):
        u_map_proj = Project('tf-eager')(self.u_map, listmode.lors)
        dx = u_map_proj.lors.data[:, 0] - u_map_proj.lors.data[:, 3]
        dy = u_map_proj.lors.data[:, 1] - u_map_proj.lors.data[:, 4]
        dz = u_map_proj.lors.data[:, 2] - u_map_proj.lors.data[:, 5]
        L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        u_map_proj = u_map_proj * L * L / self.u_map.unit_size[0] 

        return listmode / np.exp(-u_map_proj.data)


@nef_class
class UmapProject:
    def __call__(self, u_map: Umap, lors) -> Listmode:
        u_map_proj = Project('tf-eager')(u_map, lors)
        dx = u_map_proj.lors.data[:, 0] - u_map_proj.lors.data[:, 3]
        dy = u_map_proj.lors.data[:, 1] - u_map_proj.lors.data[:, 4]
        dz = u_map_proj.lors.data[:, 2] - u_map_proj.lors.data[:, 5]
        L = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        u_map_proj = u_map_proj * L * L / u_map.unit_size[0]
        return Listmode(np.exp(-u_map_proj.data), lors)
