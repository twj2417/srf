# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: __init__.py.py
@date: 4/30/2019
@desc:
'''
__all__ = ('SaveMixin', 'LoadMixin', 'save', 'load', 'DumpDataMixin', 'LoadDataMixin',
           'dump_data', 'dump_all_data', 'load_data', 'load_all_data', 'JsonDumpMixin',
           'JsonLoadMixin', 'json_dump', 'json_load')
from .local_io_mixin import SaveMixin, LoadMixin, save, load
from .data_io_mixin import DumpDataMixin, LoadDataMixin, dump_data, dump_all_data, load_data, \
    load_all_data
from .json_io_mixin import JsonDumpMixin, JsonLoadMixin, json_dump, json_load, json_dumps

# from .hashfs_io_ops import HashfsDump, HashfsLoad, hashfs_load, hashfs_dump, hashfs_dump_to_id, \
#     hashfs_load_from_id

# __all__ += ('hashfs_dump', 'hashfs_load', 'hashfs_dump_to_id', 'hashfs_load_from_id')
