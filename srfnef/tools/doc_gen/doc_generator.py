# encoding: utf-8
'''
@author: Minghao Guo
@contact: mh.guo0111@gmail.com
@software: basenef
@file: doc_generator.py
@date: 4/13/2019
@desc:
'''
import os
import sys
import time
from getpass import getuser

import matplotlib
import numpy as np
import json
from srfnef import Image, MlemFull

matplotlib.use('Agg')

author = getuser()


def title_block_gen():
    timestamp = time.time()
    datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(int(timestamp)))
    title_block = f'''
# NEF AutoDoc {datetime}
- Author: {author} 
- Generation time: {datetime}
- Operation system: {sys.platform}
- OS language: {os.environ['LANG']}
- Duration: 0.0 sec
- Total errors: 0
- Total warning: 0
- Description: 

'''
    return title_block


def _text_gen_as_table(dct: dict = {}):
    out_text = ['|key|values|\n|:---|:---|\n']
    for key, val in dct.items():
        if key == 'data':
            out_text.append(f"| {key} | Ignored |\n")
        elif not isinstance(val, dict):
            if isinstance(val, str) and len(val) > 30:
                out_text.append(f"| {key} | Ignored |\n")
            else:
                out_text.append(f"| {key} | {val} |\n")
        else:
            out_text.append(f"| {key} | {'Ignored'} |\n")

    return out_text


def json_block_gen(dct: dict = {}):
    if isinstance(dct, str):
        dct = json.loads(dct)

    dct['image_config']['size'] = np.round(dct['image_config']['size'], decimals = 3).tolist()
    if dct['emap'] is not None:
        dct['emap']['size'] = np.round(dct['emap']['size'], decimals = 3).tolist()

    json_str = json.dumps(dct, indent = 4)
    out_text = "## RECON JSON\n"
    out_text += "```javascript\n"
    out_text += json_str + '\n'
    out_text += "```\n"
    return out_text


def image_block_gen(img: Image, path: str):
    print('Generating text blocks...')
    from matplotlib import pyplot as plt
    vmax = np.percentile(img.data, 99.99)
    midind = [int(img.shape[i] / 2) for i in range(3)]
    plt.figure(figsize = (30, 10))
    plt.subplot(231)
    plt.imshow(img.data[midind[0], :, :], vmax = vmax)
    plt.subplot(232)
    plt.imshow(img.data[:, midind[1], :].transpose(), vmax = vmax)
    plt.subplot(233)
    plt.imshow(img.data[:, :, midind[2]].transpose(), vmax = vmax)

    plt.subplot(234)
    plt.plot(img.data[midind[0], midind[1], :])
    plt.subplot(235)
    plt.plot(img.data[midind[0], :, midind[2]])
    plt.subplot(236)
    plt.plot(img.data[:, midind[1], midind[2]])
    timestamp = time.time()
    datetime_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(int(timestamp)))
    plt.savefig(path + f'/out_img{datetime_str}.png')
    out_text = f'![123]({path}/out_img{datetime_str}.png)\n'
    return out_text


def statistic_block_gen(dct: dict = {}):
    out_text = []

    key_set = set()
    for name, sub_dct in dct.items():
        for key, val in sub_dct.items():
            if isinstance(val, str) and len(val) < 30:
                key_set.add(key)

    col_names = ['|name ', '|:---']
    for key in key_set:
        col_names[0] += '|' + key + ''
    else:
        col_names[0] += '|\n'
    for _ in key_set:
        col_names[1] += '|:---'
    else:
        col_names[1] += '|\n'
    out_text += col_names

    for name, sub_dct in dct.items():
        row = '| ' + name + ' '
        for key in key_set:
            if key in sub_dct:
                row += '|' + str(sub_dct[key]) + ''
            else:
                row += '|-'
        else:
            row += '|\n'

        out_text += [row]

    return out_text


def metric_block_gen(mask: np.ndarray, img: Image):
    from srfnef import image_metric as metric
    dct = {}
    # contrast hot
    dct.update(
        contrast_hot = {str(ind_): float(val_) for ind_, val_ in metric.contrast_hot(mask, img)})
    dct.update(
        contrast_cold = {str(ind_): float(val_) for ind_, val_ in metric.contrast_cold(mask, img)})
    dct.update(contrast_noise_ratio1 = metric.cnr1(mask, img))
    dct.update(contrast_noise_ratio2 = metric.cnr2(mask, img))
    dct.update(contrast_recovery_coefficiency1 = metric.crc1(mask, img))
    dct.update(contrast_recovery_coefficiency2 = metric.crc2(mask, img))
    dct.update(standard_error = metric.standard_error(mask, img))
    dct.update(normalized_standard_error = metric.nsd(mask, img))
    dct.update(standard_deviation = metric.sd(mask, img))
    dct.update(background_visibility = metric.bg_visibility(mask, img))
    dct.update(noise1 = metric.noise1(mask, img))
    dct.update(noise2 = metric.noise2(mask, img))
    dct.update(signal_noise_ratio1 = metric.snr1(mask, img))
    dct.update(signal_noise_ratio2 = metric.snr2(mask, img))
    dct.update(positive_deviation = metric.pos_dev(mask, img))
    for ind, val in dct.items():
        if not isinstance(val, dict):
            dct[ind] = float(val)
    json_str = json.dumps(dct, indent = 4)
    out_text = "## IMAGE METRIC JSON\n"
    out_text += "```javascript\n"
    out_text += json_str + '\n'
    out_text += "```\n"
    return out_text


def doc_gen(mlem_obj: MlemFull, img: Image, path: str, filename: str = None,
            mask: np.ndarray = None):
    timestamp = time.time()
    datetime_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(int(timestamp)))

    if filename is None:
        filename = 'doc_gen-' + datetime_str + '.md'
    out_text = title_block_gen()
    out_text += image_block_gen(img, path)
    out_text += json_block_gen(mlem_obj.asdict(recurse = True))
    if mask is not None:
        if isinstance(mask, str):
            mask = np.load(mask)
        out_text += metric_block_gen(mask, img)
    # out_text += statistic_block_gen(dct)
    with open(filename, 'w') as fout:
        fout.writelines(out_text)
    # print('Converting MD to PDF...')
    # import pypandoc
    # print(filename)
    # pypandoc.convert_file(filename, 'pdf', outputfile = filename + '.pdf')
    return filename
