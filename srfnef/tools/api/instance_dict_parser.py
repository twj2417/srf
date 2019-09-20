# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: srf_ct
@file: instance_dict_parser.py
@date: 4/9/2019
@desc:
'''
from srfnef import NefBaseClass
from srfnef.io import save, load


def _convert_single_instance_to_dict(obj: NefBaseClass = None, *, verbose = True):
    if obj is None:
        raise ValueError('valid instance are needed.')

    kwargs = {'classname': obj.__class__.__name__}
    for key, _type in obj.__class__.__annotations__.items():
        if not verbose and key.startswith('_'):
            continue
        if key == 'data':
            res_path = save(getattr(obj, key))
            kwargs.update({'data': res_path})  # should be file_io here
        elif issubclass(_type, (list, str, tuple, int, float, bool, type(None))):
            kwargs.update({key: getattr(obj, key)})
        else:
            kwargs.update(
                {key: _convert_single_instance_to_dict(getattr(obj, key), verbose = verbose)})

    return kwargs


def convert_instance_to_dict(objs_dct: dict, *, verbose = True):
    if isinstance(objs_dct, NefBaseClass):
        objs_dct = {str(0): objs_dct}
    elif isinstance(objs_dct, list):
        objs_dct = {str(ind): obj for ind, obj in enumerate(objs_dct)}

    kwargs = {}
    for key, obj in objs_dct.items():
        kwargs.update({key: _convert_single_instance_to_dict(obj, verbose = verbose)})
    return kwargs


def convert_dict_to_instance(dct: dict, *, schema: dict):
    if schema is None:
        raise ValueError('A valid schema is needed.')
    if isinstance(schema, str):
        import json as json_
        try:
            schema = json_.loads(schema)
        except ValueError('Can not parse schema: ', schema):
            pass
    out = {}
    for key, val in dct.items():
        if 'classname' in val:
            classname = val['classname']
            if classname not in schema:
                raise ValueError('can not find valid class assigned for', key, 'in the first arg')
            cls = schema[classname]
            print(1, cls)
            if isinstance(cls, dict):
                from .class_schema_parser import convert_schema_to_class
                cls = convert_schema_to_class(schema)[classname]
                print(2, convert_schema_to_class(schema))
        else:
            raise ValueError(f"can not find valid classname in dct.['{key}']")

        kwargs = {}
        print(cls)
        for field, type_ in cls.__annotations__.items():
            sub_ = val[field]
            if field.startswith('_'):
                continue
            elif field == 'data':
                kwargs.update({field: load(sub_)})
            elif isinstance(sub_, dict):
                kwargs.update(
                    {field: convert_dict_to_instance({field: sub_}, schema = schema)[field]})
            else:
                kwargs.update({field: type_(sub_)})
        out.update({key: cls(**kwargs)})
    return out


del NefBaseClass
