# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: srf_ct
@file: __init__.py
@date: 4/9/2019
@desc:
'''

from sqlalchemy import Column, Integer, String, ForeignKey, Float, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.dialects import postgresql

from .config import Base, create_automap_base
from srfnef.utils import convert_Camal_to_snake, convert_snake_to_Camel
from srfnef import NefBaseClass, NefBaseClass, List, NefBaseClass


def create_app_table():
    class NosqlTable(Base):
        __tablename__ = 'app_nosql_table'
        id = Column(Integer, primary_key = True)
        hash = Column(String)
        key = Column(String)
        val = Column(String, nullable = True)

    Base.metadata.create_all()


#
#
# def _auto_fill_table_class_declare(cls):
#     table_class_name = cls.__name__ + 'Table'
#     table_name = convert_Camal_to_snake(cls.__name__)
#
#     cnt = f"""
# #
# # THIS FILE IS GENERATED FROM SRFNEF/TOOLS/CREATE_TABLES.PY
# #
#
#
# class {table_class_name}:
#     __tablename__ = '{table_name}'
#     id = Column(Integer, primary_key = True)
# """
#     for key, type_ in cls.__annotations__.items():
#         if key == 'data':
#             cnt += f'    {key} = Column(String, nullable = True)\n'
#         elif type_ is int:
#             cnt += f'    {key} = Column(Integer, nullable = True)\n'
#         elif type_ is float:
#             cnt += f'    {key} = Column(Float, nullable = True)\n'
#         elif type_ is bool:
#             cnt += f'    {key} = Column(Boolean, nullable = True)\n'
#         elif type_ is str:
#             cnt += f'    {key} = Column(String, nullable = True)\n'
#         elif issubclass(type_, list):
#             if type_.dtype is float:
#                 cnt += f'    {key} = Column(postgresql.ARRAY(Float, dimensions = 1), nullable = True)\n'
#             elif type_.dtype is int:
#                 cnt += f'    {key} = Column(postgresql.ARRAY(Integer, dimensions = 1), nullable = True)\n'
#             elif type_.dtype is str:
#                 cnt += f'    {key} = Column(postgresql.ARRAY(String, dimensions = 1), nullable = True)\n'
#             elif type_.dtype is bool:
#                 cnt += f'    {key} = Column(postgresql.ARRAY(Boolean, dimensions = 1), nullable = True)\n'
#         elif issubclass(type_, (NefBaseClass, NefBaseClass)):
#             sub_table_name = type_.__name__
#             cnt += f'    {sub_table_name}_id: Column(Integer, ForeignKey({sub_table_name}.id))\n'
#             cnt += f'    {key}: relationship({type_.__name__}Table)\n'
#     cnt += f'    hash: Column(String, nullable = True)'
#
#     return cnt

# def write_to_file(cls):
#     import inspect
#     import srfnef
#     import os
#     MODULE_DIR = os.path.split(inspect.getfile(srfnef))[0] + '/'
#


def create_table_class(cls: type) -> type:
    assert issubclass(cls, NefBaseClass)

    table_class_name = cls.__name__ + 'Table'
    table_name = convert_Camal_to_snake(cls.__name__)

    kwargs = {'__tablename__': table_name, '_id': Column(Integer, primary_key = True)}

    if table_name in create_automap_base().classes:
        return getattr(create_automap_base().classes, table_name)

    for key, type_ in cls.__annotations__.items():
        if issubclass(type_, NefBaseClass):
            sub_table_class = create_table_class(type_)
            sub_table_name = sub_table_class.__tablename__
            kwargs.update({sub_table_name + '_id': Column(Integer, ForeignKey(
                sub_table_name + '._id'))})
            kwargs.update({key: relationship(type_.__name__ + 'Table')})
        elif key == 'data':
            kwargs.update({key: Column(String, nullable = True)})
        elif type_ is int:
            kwargs.update({key: Column(Integer, nullable = True)})
        elif type_ is float:
            kwargs.update({key: Column(Float, nullable = True)})
        elif type_ is bool:
            kwargs.update({key: Column(Boolean, nullable = True)})
        elif type_ is str:
            kwargs.update({key: Column(String, nullable = True)})
        elif issubclass(type_, list):
            if type_.dtype is float:
                kwargs.update(
                    {key: Column(postgresql.ARRAY(Float, dimensions = 1), nullable = True)})
            elif type_.dtype is int:
                kwargs.update(
                    {key: Column(postgresql.ARRAY(Integer, dimensions = 1), nullable = True)})
            elif type_.dtype is str:
                kwargs.update(
                    {key: Column(postgresql.ARRAY(String, dimensions = 1), nullable = True)})
            elif type_.dtype is bool:
                kwargs.update(
                    {key: Column(postgresql.ARRAY(Boolean, dimensions = 1), nullable = True)})
            else:
                raise TypeError('can not parse', type_.__name__, 'when creating a `Config` class '
                                                                 'table')
        else:
            raise TypeError('can not parse', type_.__name__, 'when creating a `Config` class table')
        kwargs.update({'__hash__': Column(String, nullable = True)})
    table_cls = type(table_class_name, (Base,), kwargs)
    Base.metadata.create_all()
    return table_cls
