# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: basenef
@file: cli.py
@date: 4/14/2019
@desc:
'''

import json

import click

from .doc_generator import doc_gen


@click.command()
@click.argument('json_path')
@click.option('--output', '-o', default = None, help = 'output file or directory')
def cli_autodoc(json_path, output):
    with open(json_path, 'r') as fin:
        dct = json.load(fin)
        out_path = doc_gen(dct, output)
    print(out_path, ' generated successful.')

# if __name__ == '__main__':
#     cli_autodoc()
