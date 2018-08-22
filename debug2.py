import os
import torch
import numpy as np
from torch.autograd import Variable
from datetime import datetime
from faster_rcnn import network
from faster_rcnn.network import init_data, data_to_variable
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.utils.timer import Timer

from faster_rcnn.roi_data_layer.sampler import sampler
from faster_rcnn.roi_data_layer.roidb import extract_roidb
from faster_rcnn.roi_data_layer.roibatchLoader import roibatchLoader
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file


# hyper-parameters
# ------------
#imdb_name = 'voc_2007_trainval'
imdb_name = 'CaltechPedestrians'
cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
#pretrained_model = 'data/pretrained_model/VGG_imagenet.npy'
pretrained_model = 'data/pretrained_model/VGGnet_fast_rcnn_iter_70000.h5'
output_dir = 'models/saved_model3'

start_epoch = 1
end_epoch = 3
lr_decay_step = 5
lr_decay = 1./10

rand_seed = 1024
_DEBUG = True
use_tensorboard = False
remove_all_log = False   # remove all historical experiments in TensorBoard
exp_name = None # the previous experiment name in TensorBoard

# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)
batch_size = cfg.TRAIN.IMS_PER_BATCH
lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval = cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS
save_interval = (cfg.TRAIN.SNAPSHOT_ITERS / batch_size)
# load data        # PASCAL VOC 2007 : Total 5011 images, 15662 objects
imdb, roidb, ratio_list, ratio_index = extract_roidb(imdb_name)
train_size = len(roidb)
sampler_batch = sampler(train_size, batch_size)
dataset = roibatchLoader(roidb, ratio_list, ratio_index, batch_size,
                                                        imdb.num_classes, training=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         sampler=sampler_batch, num_workers=0)


blob = init_data(is_cuda=True)
from time import sleep


iters_per_epoch = int(train_size / batch_size)

for epoch in range(start_epoch, end_epoch+1):
    print(epoch)
    data_iter = iter(dataloader)
    for step in range(iters_per_epoch):
        if step % 100 ==0 :
            print('{}/{}'.format(step, iters_per_epoch))
        # get one batch
        try:
            data = next(data_iter)
            (im_data, im_info, gt_boxes, num_boxes) = data_to_variable(blob, data)
        except RuntimeError as e:
            print(step)
            print(e)

