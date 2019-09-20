# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: auto_dump.py
@date: 7/8/2019
@desc:
'''

import json
from srfnef import NefBaseClass


def auto_dump(obj: NefBaseClass, tags: dict) -> str:
    dct = obj.asdict(recurse = True)
    for key, val in dct.items():
        if key == 'data' and not isinstance(val, str):
            raise ValueError('please dump data first')
    return json.dumps(dct, indent = 4)


def auto_dumps(obj: NefBaseClass, path: str = None) -> str:
    dct = obj.asdict(recurse = True)
    for key, val in dct.items():
        if key == 'data' and not isinstance(val, str):
            raise ValueError('please dump data first')
    with open(path, 'w') as fout:
        json.dump(dct, fout, indent = 4)
    return path


class AutoDump(NefBaseClass):
    def dump(self) -> str:
        return auto_dump(self)

    def dumps(self) -> str:
        return auto_dumps(self)
