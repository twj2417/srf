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
import tensorflow as tf
import json


@click.command()
@click.argument('json_path', type = click.Path(exists = True))
@click.option('--outdir', '-o', default = None)
@click.option('-s', '--system', default = 'mct')
def test_full(json_path, outdir, system):
    tf.compat.v1.enable_eager_execution()
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(json_path))
    else:
        outdir = os.path.abspath(outdir)

    if not os.path.isdir(outdir):
        os.mkdir(outdir, mode = 0o777)
    # if system == 'mct':
    mlem_full_without_data = nef.io.json_load(nef.functions.MlemFull, json_path)
    if system == '8panel':
        with open('/mnt/users/weijie/srfnef/mlem_full.json','r') as fin:
            dct = json.load(fin)
        scanner = nef.PetCylindricalScanner.from_dict(dct['scanner'])
    mlem_full_without_data = mlem_full_without_data.update(scanner = scanner)
    mlem_obj = nef.io.load_all_data(mlem_full_without_data)
    opt_atten = [None] if mlem_obj.atten_corr is None else [None, mlem_obj.atten_corr]
    opt_psf = [None] if mlem_obj.psf_corr is None else [None, mlem_obj.psf_corr]
    for atten in opt_atten:
        for psf in opt_psf:
                out_append = ''
                if atten is not None:
                    out_append += '_atten'
                if psf is not None:
                    out_append += '_psf'
                print('*************************************************')
                print(out_append)
                mlem_ = mlem_obj.update(atten_corr = atten,
                                        psf_corr = psf,
                                        scatter_corr = None)
                img = mlem_()
                from srfnef.corrections.scattering.scatter import scatter_preprocess
                scatter_preprocess(mlem_obj.scanner,mlem_obj.listmode,img,mlem_obj.atten_corr.u_map,outdir,out_append=out_append)
                nef.save(img, outdir + f'/recon_image{out_append}.hdf5')
                # with open(json_path, 'r') as fin:
                #     if 'mask' in json.load(fin):
                #         mask_path = json.load(fin)['mask']
                #     else:
                #         mask_path = None
                nef.doc_gen(mlem_full_without_data, img, outdir,
                            outdir + f'/recon_doc{out_append}.md')
                pypandoc.convert_file(outdir + f'/recon_doc{out_append}.md', 'pdf',
                                      outputfile = outdir + f'/recon_doc{out_append}.md' + '.pdf')

    return img

@click.command()
@click.argument('json_path', type = click.Path(exists = True))
@click.option('--outdir', '-o', default = None)
@click.option('-s', '--system', default = 'mct')
def test_with_scatter(json_path, outdir,system):
    tf.enable_eager_execution()
    if outdir is None:
        outdir = os.path.dirname(os.path.abspath(json_path))
    else:
        outdir = os.path.abspath(outdir)

    if not os.path.isdir(outdir):
        os.mkdir(outdir, mode = 0o777)

    mlem_full_without_data = nef.io.json_load(nef.functions.MlemFull, json_path)
    if system == '8panel':
        with open('/mnt/users/weijie/srfnef/mlem_full.json','r') as fin:
            dct = json.load(fin)
        scanner = nef.PetCylindricalScanner.from_dict(dct['scanner'])
    mlem_full_without_data = mlem_full_without_data.update(scanner = scanner)
    mlem_obj = nef.io.load_all_data(mlem_full_without_data)
    opt_atten = [None] if mlem_obj.atten_corr is None else [None, mlem_obj.atten_corr]
    opt_psf = [None] if mlem_obj.psf_corr is None else [None, mlem_obj.psf_corr]
    for atten in opt_atten:
        for psf in opt_psf:               
            out_append = ''
            if atten is not None:
                out_append += '_atten'
            if psf is not None:
                out_append += '_psf'
            print('*************************************************')  
            print("scattering")         
            img = nef.load(nef.Image,outdir + f'/recon_image{out_append}.hdf5')          
            listmode = mlem_obj.scatter_corr(img,mlem_obj.atten_corr.u_map,mlem_obj.scanner,outdir,out_append)    
            mlem_ = nef.MlemFull(mlem_obj.n_iter, mlem_obj.image_config, mlem_obj.scanner, listmode)
            img = mlem_()
            out_append += '_scatter'
            print(out_append)
            nef.save(img, outdir + f'/recon_image{out_append}.hdf5')
            nef.doc_gen(mlem_full_without_data, img, outdir,
                    outdir + f'/recon_doc{out_append}.md')
            pypandoc.convert_file(outdir + f'/recon_doc{out_append}.md', 'pdf',
                        outputfile = outdir + f'/recon_doc{out_append}.md' + '.pdf')

    return img


if __name__ == '__main__':
    test_full()
