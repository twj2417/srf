# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: multi_bed_merge.py
@date: 5/30/2019
@desc:
'''
import numpy as np
from srfnef import nef_class, Image, List, Emap, PetEcatScanner, ImageConfig, \
    Listmode, Lors
from srfnef.functions import EmapGenerator
from srfnef.ops.deform_mixins import DeformMixin
from copy import copy


@nef_class
class ImageMerger(DeformMixin):
    def __call__(self, images: List(Image), fov_centers: List(float)) -> Image:
        '''sort images according to their fov centers'''
        if len(images) == 1:
            return images[0]
        border_left = np.min([image.center[2] - image.size[2] / 2 for image in images])
        border_right = np.max([image.center[2] + image.size[2] / 2 for image in images])
        unit_size_z = images[0].unit_size[2]
        length = border_right - border_left
        shape_z = int(np.round(length / unit_size_z))
        size_z = shape_z * unit_size_z
        center_z = (border_left + border_right) / 2

        shape = images[0].shape[:2] + [shape_z]
        center = images[0].center[:2] + [center_z]
        size = images[0].size[:2] + [size_z]

        trans = (np.array(fov_centers[1:]) + np.array(fov_centers[:-1])) / 2
        trans_ind = np.round((trans - center[2] + size[2] / 2) / 2.05).astype(int).tolist()
        trans_ind = [0] + trans_ind + [-1]
        print(trans_ind)
        img_out = Image(np.zeros(shape, np.float32), center, size)
        img_out_data = img_out.data
        for ind, img in enumerate(images):
            img_data = copy(img.data)
            img_data[:, :, :trans_ind[ind]] = 0
            img_data[:, :, trans_ind[ind + 1]:] = 0
            img_out_data += img_data
        return img_out.update(data = img_out_data)


@nef_class
class EmapMerger:
    def __call__(self, scanner: PetEcatScanner,
                 config: ImageConfig,
                 fov_centers: List(float)) -> Emap:
        emap = Emap(np.zeros(config.shape, np.float32), config.center, config.size)
        for i in range(len(fov_centers)):
            scanner = scanner.update(center = [0, 0, fov_centers[i]])
            emap += EmapGenerator('block-full', scanner)(emap)
        return emap


@nef_class
class ListmodeMerger:
    def __call__(self, listmodes: List(Listmode)) -> Listmode:
        lors = Lors(np.vstack([lm.lors.data for lm in listmodes]))
        listmode = Listmode(np.hstack([lm.data for lm in listmodes]), lors)
        return listmode
