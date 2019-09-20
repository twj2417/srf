# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: nef
@file: app_recon_full.py
@date: 5/13/2019
@desc:
'''
import click
import srfnef as nef
import os
import pypandoc


@click.command()
@click.argument('json_path', type = click.Path(exists = True))
@click.option('--outdir', '-o', default = None)
def recon_full(json_path, outdir):
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(json_path))
    else:
        outdir = os.path.abspath(outdir)

    if not os.path.isdir(outdir):
        os.mkdir(outdir, mode = 0o777)

    mlem_full_without_data = nef.io.json_load(nef.functions.MlemFullCylindrical, json_path)
    mlem_obj = nef.io.load_all_data(mlem_full_without_data)
    img = mlem_obj()
    nef.save(img, outdir + '/recon_image.hdf5')
    nef.save(mlem_full_without_data, outdir + '/mlem_full.hdf5')
    nef.doc_gen(mlem_full_without_data, img, outdir, outdir + '/recon_doc.md')
    pypandoc.convert_file(outdir + '/recon_doc.md', 'pdf',
                          outputfile = outdir + '/recon_doc.md' + '.pdf')
    return img


if __name__ == '__main__':
    recon_full()
