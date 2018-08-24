import os
import torch
import cv2
import numpy as np

from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.faster_rcnn2 import FasterRCNN
from faster_rcnn import network
from faster_rcnn.roi_data_layer.roidb import extract_roidb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file
cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
cfg_from_file(cfg_file)
pretrained_model = 'data/pretrained_model/resnet50_imagenet.pth'
imdb_name = 'CaltechPedestrians'
imdb = get_imdb(imdb_name)
_DEBUG = True
net = FasterRCNN(classes=imdb.classes, debug=_DEBUG)
network.weights_normal_init(net, dev=0.01)





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
