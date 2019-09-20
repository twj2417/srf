# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: srf_ct
@file: json.py
@date: 4/9/2019
@desc:
'''
import json as json_

from srfnef import NefClass
from .class_schema_parser import convert_class_to_schema, convert_schema_to_class
from .instance_dict_parser import convert_instance_to_dict, convert_dict_to_instance


def loads(json_string: str, *, schema = None):
    dct = json_.loads(json_string)
    is_class = 'classname' not in list(dct.values())[0]
    if not is_class:  # instance
        return convert_dict_to_instance(dct, schema = schema)
    else:
        return convert_schema_to_class(dct)


def load(path: str, *, schema = None):
    with open(path, 'r') as fin:
        dct = json_.load(fin)
    is_class = 'classname' not in list(dct.values())[0]
    if not is_class:  # instance
        return convert_dict_to_instance(dct, schema = schema)
    else:
        return convert_schema_to_class(dct)


def dumps(o: dict, *, verbose = False):
    if isinstance(o, NefClass):
        o = {'0': o}
        is_class = False
    elif isinstance(o, type):
        o = {o.__name__: o}
        is_class = True
    elif isinstance(o, list):
        if not o:
            return ''
        elif isinstance(o[0], type):
            o = {oo.__name__: oo for oo in o}
            is_class = True
        else:
            o = {ind: oo for ind, oo in enumerate(o)}
            is_class = False
    elif isinstance(o, dict):
        if not o:
            return ''
        elif isinstance(list(o.values())[0], type):
            is_class = True
        else:
            is_class = False
    else:
        raise NotImplementedError

    if not is_class:  # instance
        dct = convert_instance_to_dict(o, verbose = verbose)
    else:
        dct = convert_class_to_schema(o, verbose = verbose)
    return json_.dumps(dct)


def dump(o: dict, path: str, *, verbose = False):
    if isinstance(o, NefClass):
        o = {'0': o}
        is_class = False
    elif isinstance(o, type):
        o = {o.__name__: o}
        is_class = True
    elif isinstance(o, list):
        if not o:
            return ''
        elif isinstance(o[0], type):
            o = {oo.__name__: oo for oo in o}
            is_class = True
        else:
            o = {ind: oo for ind, oo in enumerate(o)}
            is_class = False
    elif isinstance(o, dict):
        if not o:
            return ''
        elif isinstance(list(o.values())[0], type):
            is_class = True
        else:
            is_class = False
    else:
        raise NotImplementedError

    if not is_class:  # instance
        dct = convert_instance_to_dict(o, verbose = verbose)
    else:
        dct = convert_class_to_schema(o, verbose = verbose)
    with open(path, 'w') as fout:
        json_.dump(dct, fout, indent = 4, separators = [',', ':'])


del json_
