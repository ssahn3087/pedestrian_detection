import os
import json
import time
import numpy as np
import numpy.random as npr
from collections import Counter
global coco_set
global save_name
"""
    USE: coco/annotations/make_annotations.py
        extract dataset for training by making whole set into a small set
        train : 13k images
        val  : 3k images
        * First change the name of original file ( containing entire dataset )
        ex) instances_train_2017(default) -> instances_train_2017_all.json
        Then code will make a small dataset named 'instances_train_2017'
"""
coco_set = 'instances_train2017_all.json'
save_name = '_'.join(coco_set.split('_')[:-1]) + '.json'
# 20 classes
post_class = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'boat', 'train', 'airplane',
              'truck', 'bird', 'cat',  'dog', 'horse', 'sheep', 'cow',
              'fire hydrant', 'stop sign', 'traffic light', 'parking meter', 'bench']
#for train_set
sub_objs = 10000
main_objs = 8000
min = 1200
#for val[test]_set
# sub_objs = 4000
# main_objs = 3000
# min = 500

def extract_by_post_class(post_class):
    area_threshold = 200.0
    print('Area threshold exists : ', area_threshold)
    dataset = json.load(open(coco_set, 'r'))
    for k,v in dataset.items():
        print(k, len(v))
    t = time.time()
    need_id = []
    categories = []
    Count = {}
    for cat in dataset['categories']:
        if cat['name'] in post_class:
            Count[cat['id']] = 0
            need_id.append(cat['id'])
            categories.append(cat)
    dataset['categories'] = categories

    print('Done : {}s'.format(time.time() - t))
    annotations = []
    annotations_im = []
    image_id = []
    image_id_im = []

    for ann in dataset['annotations']:
        if ann['area'] <= area_threshold:
            continue
        # take person and car class apart
        if ann['category_id'] == 1 or ann['category_id'] == 3:
            annotations_im.append(ann)
            image_id_im.append(ann['image_id'])
        elif ann['category_id'] in need_id:
            if Count[ann['category_id']] < min:
                Count[ann['category_id']] += 1
                annotations.append(ann)
                image_id.append(ann['image_id'])
    if main_objs > 0 :
        indices = np.sort(npr.choice(len(annotations_im), main_objs, replace=False))
        annotations_im = np.asarray(annotations_im)[indices].tolist()
        image_id_im = np.asarray(image_id_im)[indices].tolist()
    if sub_objs > 0:
        indices = np.sort(npr.choice(len(annotations), sub_objs, replace=False))
        annotations = np.asarray(annotations)[indices].tolist()
        image_id = np.asarray(image_id)[indices].tolist()
    annotations.extend(annotations_im)
    image_id.extend(image_id_im)

    image_id = list(set(image_id))

    dataset['annotations'] = annotations
    print('Done : {}s'.format(time.time() - t))
    images = []
    for image in dataset['images']:
        if image['id'] in image_id:
            images.append(image)
    dataset['images'] = images
    print('Done : {}s'.format(time.time() - t))
    with open(save_name, 'w') as outfile:
        json.dump(dataset, outfile)
    print('new annotation file saved : ',save_name)


if __name__ == '__main__':
    extract_by_post_class(post_class)