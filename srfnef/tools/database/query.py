# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: srf_ct
@file: query.py
@date: 3/13/2019
@desc:
'''
from .config import create_session, create_automap_base


def query_fields_with_id(id_: int = -1, table_name: str = ''):
    Base = create_automap_base()
    TableClass = getattr(Base.classes, table_name)

    with create_session() as session:
        table_instance = session.query(TableClass).filter(TableClass._id == id_).all()[0]
        out_dct = {}
        for key, val in table_instance.__dict__.items():
            if key.startswith('_'):
                continue
            elif key in getattr(TableClass, 'foreign_keys', lambda x = 1: [])():
                out_dct.update({key: query_fields_with_id(val, key)})
            else:
                out_dct.update({key: val})
    return out_dct
