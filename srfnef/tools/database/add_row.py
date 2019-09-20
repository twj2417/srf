# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: srf_ct
@file: add_objects.py
@date: 4/10/2019
@desc:
'''

import hashlib
from srfnef.utils import convert_Camal_to_snake
from .config import create_session, create_automap_base


def add_dicts(dct: dict):
    assert '__classname__' in dct
    table_name = convert_Camal_to_snake(dct['__classname__'])
    Base = create_automap_base()
    TableClass = Base.classes[table_name]
    kwargs = {}
    m = hashlib.sha256()

    for key, val in dct.items():
        if key.startswith('__'):
            continue
        elif isinstance(val, dict):
            print(val)
            sub_table_instance, sub_hash = add_dicts(val)
            print(sub_table_instance)
            kwargs.update({key: sub_table_instance})
            m.update(str(sub_hash).encode('utf-8'))
        else:
            kwargs.update({key: val})
            m.update(str(val).encode('utf-8'))

    hash_ = 'sha256:' + m.hexdigest()
    kwargs.update({'__hash__': hash_})
    print(kwargs)
    ''' hash check '''
    table_instance = TableClass(**kwargs)

    with create_session() as session:
        ans = session.query(TableClass.__hash__).filter(TableClass.__hash__ == hash_).all()

        if not ans:
            session.add(table_instance)
            session.commit()
        else:
            pass
    return table_instance, hash_
