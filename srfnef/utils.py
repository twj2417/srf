# encoding: utf-8
'''
srfnef.utils
~~~~~~~~~~~~

This module provides utility functions that are used within SRF-NEF
that are alose useful for extenel comsumptions.
'''

import hashlib
import os
import platform
import re
import sys

import tqdm as tqdm_

# from dxl.core.config import ConfigProxy, CNode, Configuration

DEFAULT_CONFIGURATION_NAME = 'SRF_CONFIGURATION'

__all__ = (
    'is_notebook', 'tqdm', '_eps', '_small', '_tiny', '_huge', '_pi', 'main_path', 'separator',
    'declare_eager_execution', 'clear_gpu')


def is_notebook():
    '''check if the current environment is `ipython`/ `notebook`
    '''
    return 'ipykernel' in sys.modules


is_ipython = is_notebook


def tqdm(*args, **kwargs):
    '''same as tqdm.tqdm
    Automatically switch between `tqdm.tqdm` and `tqdm.tqdm_notebook` accoding to the runtime
    environment.
    '''
    if is_notebook():
        return tqdm_.tqdm_notebook(*args, **kwargs)
    else:
        return tqdm_.tqdm(*args, **kwargs)


_eps = 1e-8

_small = 1e-4

_tiny = 1e-8

_huge = 1e8

_pi = 3.14159265358979323846264338

if 'Windows' in platform.system():
    separator = '\\'
else:
    separator = '/'

main_path = os.path.abspath(os.path.dirname(
    os.path.abspath(__file__))) + separator


def convert_Camal_to_snake(name: str) -> str:
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def convert_snake_to_Camel(name: str) -> str:
    out = ''
    for ele in name.split('_'):
        out += ele.capitalize()
    return out


def get_hash_of_timestamp():
    import time
    m = hashlib.sha256()
    timestamp = time.time()
    m.update(str(timestamp).encode('utf-8'))
    return m.hexdigest()


def file_hasher(path: str) -> str:
    import os
    if os.path.isdir(path):
        raise ValueError('Only file can be hashed')

    BLOCKSIZE = 65536
    m = hashlib.sha256()

    with open(path, 'rb') as fin:
        buf = fin.read(BLOCKSIZE)
        while len(buf) > 0:
            m._update(buf)
            buf = fin.read(BLOCKSIZE)
    return m.hexdigest()


def declare_eager_execution():
    import tensorflow as tf

    if not tf.compat.v1.executing_eagerly():
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        tf.compat.v1.enable_eager_execution()
        # tf.compat.v1.enable_eager_execution(config = config)


def clear_gpu(ind = 0):
    from numba import cuda
    cuda.select_device(ind)
    cuda.close()
    cuda.select_device(ind)
#
#
# def config_with_name(name):
#     if ConfigProxy().get(DEFAULT_CONFIGURATION_NAME) is None:
#         ConfigProxy()[DEFAULT_CONFIGURATION_NAME] = Configuration(CNode({}))
#     return ConfigProxy().get_view(DEFAULT_CONFIGURATION_NAME, name)
#
#
# def clear_config():
#     ConfigProxy.reset()
