# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: module_parser.py
@date: 4/26/2019
@desc:
'''
import inspect
import json

import srfnef as nef
from srfnef import NoneType


def _class_name(cls):
    if isinstance(cls, tuple) and len(cls) == 2 and cls[1] == NoneType:
        return cls[0].__name__
    elif isinstance(cls, type):
        return cls.__name__
    else:
        raise TypeError('can not find valid name for', cls.__name__)


def module_parser(mod, *, max_deps = 5, module_name = None,
                  exclude = None,
                  include = None,
                  superclass = None):
    if module_name is None:
        module_name = mod.__name__
    out_dct = {}
    if max_deps == 0 or mod is None:
        return out_dct
    for key, val in inspect.getmembers(mod):
        if key in out_dct:
            continue
        if key.startswith('__'):
            continue
        elif exclude is not None and key in exclude:
            continue

        if inspect.isclass(val):
            if superclass is not None and not issubclass(val, superclass):
                continue

            if hasattr(val, 'keys'):
                out_dct.update(
                    {key: {name: _class_name(tp) for name, tp in zip(val.keys(), val.types())}})
            else:
                out_dct.update({key: {}})
            if hasattr(val, '__call__'):
                sig = inspect.signature(val.__call__)
                if not str(sig) == '(self, *args, **kwargs)':
                    sig_out = {k: v for k, v in val.__call_annotations__().items()}
                    out_dct[key].update({'__call__': str(sig_out)})
            else:
                out_dct.update({key: val})


        # elif inspect.isfunction(val):
        #     out_dct.update({key: str(inspect.signature(val))})
        elif inspect.ismodule(val):
            if module_name not in val.__name__:
                continue
            sub_dct = module_parser(val, max_deps = max_deps - 1,
                                    module_name = module_name,
                                    exclude = exclude,
                                    include = include,
                                    superclass = superclass)
            if sub_dct:
                if include is None or key in include:
                    out_dct.update(sub_dct)

    return out_dct
