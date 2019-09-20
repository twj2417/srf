# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: hashfs_io_ops.py
@date: 6/6/2019
@desc:
'''
from dxl.cluster.hashfs.base import PiClusterHashFs
from srfnef import nef_class, NefBaseClass
from .json_io_mixin import json_dump, json_load
from .data_io_mixin import load_all_data
import os
import deepdish as dd
import json


@nef_class
class HashfsDump:
    def _hashfs_dump_data(self, obj: NefBaseClass) -> NefBaseClass:
        out_dct = {}
        for key, val in obj.items():
            if key == 'data':
                dd.io.save('.temp.hdf5', val, compression = None)
                out_dir = PiClusterHashFs.put(_file = '.temp.hdf5', _type = [], tags = [],
                                              commit = True)
                out_dct.update({key: out_dir['url']})
                os.remove('.temp.hdf5')
            elif isinstance(val, NefBaseClass):
                out_dct.update({key: self._hashfs_dump_data(val)})
        return obj.update(**out_dct)

    def hashfs_dump(self, obj: NefBaseClass):
        obj_without_data = self._hashfs_dump_data(obj)
        json_dump(obj_without_data, '.temp.json')
        out_dir = PiClusterHashFs.put(_file = '.temp.json', _type = [], tags = [],
                                      commit = True)
        return out_dir['url']


@nef_class
class HashfsLoad:
    def hashfs_load(self, cls: type, path: str) -> NefBaseClass:
        obj_without_data = json_load(cls, path)
        return load_all_data(obj_without_data)


hashfs_load = HashfsLoad().hashfs_load
hashfs_dump = HashfsDump().hashfs_dump

json_id_file = os.environ['HOME'] + '/temp_hashfs_id.json'


@nef_class
class HashfsDumpToId:
    def hashfs_dump_to_id(self, obj: NefBaseClass) -> int:
        path_ = hashfs_dump(obj)
        if not os.path.exists(json_id_file):
            with open(json_id_file, 'w+') as fout:
                json.dump({}, fout)
        with open(json_id_file, 'r') as fin:
            dct = json.load(fin)

        if not dct.keys():
            curr_id = 0
        else:
            curr_id = 1 + max([int(i) for i in dct.keys()])
        dct.update({curr_id: path_})
        with open(json_id_file, 'w') as fout:
            json.dump(dct, fout)

        return curr_id


hashfs_dump_to_id = HashfsDumpToId().hashfs_dump_to_id


@nef_class
class HashfsLoadFromId:
    def hashfs_load_from_id(self, cls: type, id: int) -> NefBaseClass:
        with open(json_id_file, 'r') as fin:
            path_ = json.load(fin)[str(id)]
        return hashfs_load(cls, path_)


hashfs_load_from_id = HashfsLoadFromId().hashfs_load_from_id
