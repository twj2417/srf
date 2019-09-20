# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: recon_simpe.py
@date: 5/28/2019
@desc:
'''
import click
import srfnef as nef
import os
import pypandoc


@click.command()
@click.argument('listmode_path', type = click.Path(exists = True))
@click.argument('outdir', type = click.Path())
@click.option('-s', '--system', default = 'mct', help = 'existing system name, e.g. mct')
@click.option('-n', '--n_iter', default = 15, help = 'iterative number')
def recon_simple(listmode_path, outdir, system, n_iter):
    listmode = nef.load(nef.Listmode, listmode_path)
    if system == 'mct':
        image_config = nef.image_config_mct
        scanner = nef.scanner_mct
    else:
        raise NotImplementedError
    mlem_full = nef.MlemFull(n_iter, image_config, scanner, listmode)
    mlem_full_without_data = mlem_full.update(listmode = listmode_path)
    img = mlem_full()
    if outdir == '.':
        pass
    elif not os.path.isdir(outdir):
        os.mkdir(outdir, mode = 0o777)
    nef.save(img, outdir + '/recon_image.hdf5')
    nef.save(mlem_full, outdir + '/mlem_full.hdf5')

    doc_path = nef.doc_gen(mlem_full_without_data, img, outdir, outdir + '/recon_doc.md')
    print('generating doc at', doc_path)
    pypandoc.convert_file(outdir + '/recon_doc.md', 'pdf',
                          outputfile = outdir + '/recon_doc.md' + '.pdf')
    return img


if __name__ == '__main__':
    recon_simple()
