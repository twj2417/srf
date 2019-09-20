# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: srf_ct
@file: cli.py
@date: 3/4/2019
@desc:
'''
import click


#
#
# @click.command()
# @click.option('-m', '--mode', type = click.Choice(['auto', 'generate']), default = 'auto')
# @click.option('-f', '--function', type = click.Path(exists = True), help = 'filetion file path, '
#                                                                            'e.g., ./mlem.hdf5')
# @click.option('-i', '--input', type = click.Path(exists = True), help = 'input file path, e.g., '
#                                                                         './listmode.hdf5')
# @click.option('-o', '--output', type = click.Path(), help = 'output file path, e.g, '
#                                                             './recon_image.hdf5', default =
#               './result.hdf5')
# def cli(mode, function, input, output):
#     import srfnef as nef
#     if mode == 'auto':
#         func = nef.load(click.format_filename(function))
#         click.echo('Running ' + func.split(('.', '/')[-2]) + ' process...')
#         _in = nef.load(click.format_filename(input))
#         _out = func(_in)
#         nef.save(click.format_filename(output), _out)
#     elif mode == 'generate':
#         pass
#
#
# if __name__ == '__main__':
#     cli()
