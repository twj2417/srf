# encoding: utf-8
"""
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: srf_ct
@file: geometry.py
@date: 3/13/2019
@desc:
"""
from contextlib import contextmanager
from sqlalchemy.orm import sessionmaker

ws1_engine_url = 'postgresql://postgres:postgres@192.168.1.111/nef_db_new'


def create_engine(engine_url = ws1_engine_url):
    from sqlalchemy import create_engine
    engine = create_engine(engine_url, echo = False)
    return engine


@contextmanager
def create_session(*, engine_url = ws1_engine_url, is_commit = True):
    engine = create_engine(engine_url)
    Session = sessionmaker(bind = engine)
    session = Session()
    try:
        yield session
    finally:
        if is_commit:
            session.commit()
        session.close()


from sqlalchemy.ext.declarative import declarative_base

engine = create_engine(ws1_engine_url)
Base = declarative_base()
Base.metadata.bind = engine


def create_automap_base(engine_url = ws1_engine_url):
    from sqlalchemy.ext.automap import automap_base
    Base = automap_base()
    engine = create_engine(engine_url)
    Base.prepare(engine, reflect = True)
    return Base
