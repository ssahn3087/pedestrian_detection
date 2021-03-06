#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Martin Kersner, m.kersner@gmail.com
# 2016/12/08

# Modied version from https://github.com/mitmul/caltech-pedestrian-dataset-converter

import os
import sys
import glob
import json
import numpy as np
from scipy.io import loadmat
from collections import defaultdict

# file structure
# ---annotations (set00 - 10)
#    code (original matlab code)
#    seq_files   (set00 - 10)
#    extract_annotations.py && extract_images.py
def make_dir(dir_path):
	if not os.path.exists(dir_path):
		os.makedirs(dir_path)
if len(sys.argv) != 3:
    print ("Usage:")
    print ("python extract_annotations.py path/annotations output_file.json")
    exit()

annotation_path = os.path.join(os.getcwd(), 'extracted/annotations')
make_dir(annotation_path)
all_obj = 0
data = defaultdict(dict)

for dname in sorted(glob.glob(os.path.join(sys.argv[1], 'set*'))):
    set_name = os.path.basename(dname)
    data[set_name] = defaultdict(dict)

    for anno_fn in sorted(glob.glob('{}/*.vbb'.format(dname))):
        vbb = loadmat(anno_fn)

        nFrame   = int(vbb['A'][0][0][0][0][0])
        objLists = vbb['A'][0][0][1][0]
        maxObj   = int(vbb['A'][0][0][2][0][0])
        objInit  = vbb['A'][0][0][3][0]
        objLbl   = [str(v[0]) for v in vbb['A'][0][0][4][0]]
        objStr   = vbb['A'][0][0][5][0]
        objEnd   = vbb['A'][0][0][6][0]
        objHide  = vbb['A'][0][0][7][0]
        altered  = int(vbb['A'][0][0][8][0][0])
        log      = vbb['A'][0][0][9][0]
        logLen   = int(vbb['A'][0][0][10][0][0])

        video_name = os.path.splitext(os.path.basename(anno_fn))[0]

        data[set_name][video_name]['nFrame']  = nFrame
        data[set_name][video_name]['maxObj']  = maxObj
        data[set_name][video_name]['log']     = log.tolist()
        data[set_name][video_name]['logLen']  = logLen
        data[set_name][video_name]['altered'] = altered
        data[set_name][video_name]['frames']  = defaultdict(list)

        n_obj = 0
        for frame_id, obj in enumerate(objLists):
            if len(obj) > 0:
                for id, pos, occl, lock, posv in zip(
                        obj['id'][0], obj['pos'][0], obj['occl'][0],
                        obj['lock'][0], obj['posv'][0]):
                    keys = obj.dtype.names
                    id   = int(id[0][0]) - 1
                    pos  = pos[0].tolist()
                    occl = int(occl[0][0])
                    lock = int(lock[0][0])
                    posv = posv[0].tolist()
                    datum = dict(zip(keys, [id, pos, occl, lock, posv]))

                    datum['lbl']  = str(objLbl[datum['id']])
                    datum['str']  = int(objStr[datum['id']]) - 1
                    datum['end']  = int(objEnd[datum['id']]) - 1
                    datum['hide'] = int(objHide[datum['id']])
                    datum['init'] = int(objInit[datum['id']])
                    data[set_name][video_name]['frames'][frame_id].append(datum)
                    n_obj += 1

        print(dname, anno_fn, n_obj)
        all_obj += n_obj

print('Number of objects:', all_obj)
json.dump(data, open(os.path.join(annotation_path,sys.argv[2]), 'w'))
