import os
import torch
import cv2
import numpy as np

from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.roi_data_layer.roidb import extract_roidb
from faster_rcnn.fast_rcnn.config import cfg

pretrained_model = 'data/pretrained_model/VGGnet_fast_rcnn_iter_70000.h5'
imdb_name = 'CaltechPedestrians'
imdb = get_imdb(imdb_name)
index = '2601/1053/set00/V001'
data = imdb._load_pedestrian_annotation(index)
js = imdb.annotations
#imdb, roidb, ratio_list, ratio_index = extract_roidb(imdb_name)
import PIL
#print('CaltechPedestrians dataset has {} images in total, Max per episode {} images'\
#                                    .format(i, self.scene_per_episode_max))
def take_boxes(objs, imdb):
    # Check abnormal data and Remove
    objs = [obj for obj in objs if imdb.object_condition_satisfied(obj, index)]

    num_objs = len(objs)
    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    if not num_objs == 0:
        for ix, obj in enumerate(objs):
            pos = np.round(np.array(obj['pos'], dtype=np.float32))
            label = obj['lbl']
            # Make pixel indexes 0-based
            l = pos[0] - 1
            t = pos[1] - 1
            w = pos[2]
            h = pos[3]
            # boxes, gt_classes, seg_areas, ishards, overlaps
            boxes[ix, :] = [l, t, l + w, t + h]
    return boxes

path = imdb.image_path_from_index(index)
episodes = imdb.get_epsiode()
image_index = imdb.image_index

for ep, v in episodes.items():
    v = np.array(v, dtype=np.int32)
    (min_v, max_v) = str(min(v)), str(max(v))
    print(ep , min_v, max_v)
    (set_name, video_name) = ep[0], ep[1]
    objs = js[set_name][video_name]["frames"][min_v]
    boxes = take_boxes(objs, imdb)
    img1 = "{}/{}/{}/{}{}".format(imdb.image_path, set_name, video_name, min_v, imdb._image_ext)
    im1 = cv2.imread(img1, cv2.IMREAD_COLOR)
    for i in range(len(boxes)):
        (x1, y1, x2, y2) = boxes[i, 0:4]
        if not (np.array((x1, y1, x2, y2), dtype=np.int) == 0).all():

            im1 = cv2.rectangle(im1, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.imshow(str(min_v), im1)
    cv2.waitKey(500)
    img2 = "{}/{}/{}/{}{}".format(imdb.image_path, set_name, video_name, max_v, imdb._image_ext)
    im2 = cv2.imread(img2, cv2.IMREAD_COLOR)
    objs = js[set_name][video_name]["frames"][max_v]
    boxes = take_boxes(objs, imdb)
    for i in range(len(boxes)):
        (x1, y1, x2, y2) = boxes[i, 0:4]
        if not (np.array((x1, y1, x2, y2), dtype=np.int) == 0).all():

            im2 = cv2.rectangle(im2, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.imshow(str(max_v), im2)
    cv2.waitKey(2000)
    cv2.destroyAllWindows()
#print(data)


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net_pedestrians(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    cls_related_part = list(net.state_dict().keys())[-4:]
    own_dict = net.state_dict()
    need_index = [0, 15]
    # num_classses = size of score_fc.fc.bias
    num_classes = len(list(net.state_dict().values())[-3])
    irrelvant_indices = np.where(np.isin(np.arange(num_classes), need_index) \
                                 == False)[0]
    for k, v in own_dict.items():
        data = np.asarray(h5f[k])
        if k in cls_related_part:
            if (str(k).startswith('score')):
                data[irrelvant_indices] = 0.
            elif (str(k).startswith('bbox')):
                data = data.reshape(num_classes, 4, -1)
                data[irrelvant_indices] = 0.
                data = data.reshape(num_classes * 4, -1)
        param = torch.from_numpy(data)
        v.copy_(param)



"""
h5f = h5py.File(pretrained_model, mode='r')
net = FasterRCNN()
cls_related_part = list(net.state_dict().keys())[-4:]
own_dict = net.state_dict()
need_index = [0, 15]
num_classes = imdb.num_classes
for k, v in own_dict.items():
    if k in cls_related_part:
        data = np.asarray(h5f[k])
        irrelvant_indices = np.where(np.isin(np.arange(num_classes), need_index) \
                                     == False)[0]
        if(str(k).startswith('score')):
            data[irrelvant_indices] = 0.
            print(k, data[need_index])
        elif(str(k).startswith('bbox')):
            data = data.reshape(num_classes, 4, -1)
            data[irrelvant_indices] = 0.
            data = data.reshape(num_classes * 4, -1)
    param = torch.from_numpy(data)
    v.copy_(param)

# new_zeros(size, dtype=None, device=None, requires_grad=False) → Tensor
# network.load_net(pretrained_model, net)
"""
