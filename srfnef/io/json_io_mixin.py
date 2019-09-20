# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: json_io_mixin.py
@date: 5/9/2019
@desc:
'''
import json
from srfnef import NefBaseClass

from srfnef.utils import get_hash_of_timestamp


def json_dumps(obj: NefBaseClass) -> str:
    dct = obj.asdict(recurse = True)
    for key, val in dct.items():
        if key == 'data' and not isinstance(val, str):
            raise ValueError('please dump data first')
    return json.dumps(dct, indent = 4)


def json_dump(obj: NefBaseClass, path: str = None) -> str:
    if path is None:
        path = get_hash_of_timestamp() + '.json'
    elif path.endswith('.json'):
        pass
    else:
        path = path + '/' + get_hash_of_timestamp() + '.json'
    dct = obj.asdict(recurse = True)
    for key, val in dct.items():
        if key == 'data' and not isinstance(val, str):
            raise ValueError('please dump data first')
    with open(path, 'w') as fout:
        json.dump(dct, fout, indent = 4)
    return path


class JsonDumpMixin(NefBaseClass):
    def json_dump(self) -> str:
        return json_dump(self)

    def json_dumps(self) -> str:
        return json_dumps(self)


def json_load(cls: type, path: str) -> NefBaseClass:
    with open(path, 'r') as fin:
        dct = json.load(fin)

    return cls.from_dict(dct)


def json_loads(cls: type, json_str: str) -> NefBaseClass:
    dct = json.loads(json_str)

    kwargs = {}
    for key, type_ in cls.__annotations__.items():
        if key not in dct:
            kwargs.update({key: None})
        elif isinstance(type_, NefBaseClass):
            kwargs.update({key: json_load(type_, dct[key])})
        else:
            kwargs.update({key: dct[key]})
    return cls.from_dict(kwargs)


class JsonLoadMixin(NefBaseClass):
    @classmethod
    def json_loads(cls, json_str: str) -> NefBaseClass:
        return json_loads(cls, json_str)

    @classmethod
    def json_load(cls, path: str) -> NefBaseClass:
        return json_load(cls, path)
