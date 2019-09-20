# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: atom_listmode_merge.py
@date: 7/2/2019
@desc:
'''
import click
import srfnef as nef
from srfnef.adapters.listmode_from_gate_out import listmode_from_gate_out
import os


@click.command()
@click.argument('path', type = click.Path(exists = True))
@click.argument('nsub')
@click.argument('outpath', type = click.Path())
@click.option('-s', '--system', default = 'mct')
def extract_listmode(path, nsub, outpath, system):
    path = path.replace('sub.0', 'sub.?')
    if system == 'mct':
        listmode = listmode_from_gate_out(path, nef.scanner_mct, int(nsub))
    else:
        raise NotImplementedError

    if not outpath.endswith('.hdf5'):
        outdir = outpath
        if not os.path.isdir(outdir):
            os.mkdir(outdir, mode = 0o777)
        outpath = outdir + '/listmode_trans.hdf5'
    else:
        pass
    nef.save(listmode, outpath)
    print(outpath)


if __name__ == '__main__':
    extract_listmode()
