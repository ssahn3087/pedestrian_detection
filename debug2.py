import os
import torch
import numpy as np
import cv2
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
dataset = roibatchLoader(imdb, roidb, ratio_list, ratio_index, batch_size,
                                                        imdb.num_classes, training=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         sampler=sampler_batch, num_workers=0)

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
            img_id = data[4]
            im_info = data[1]
            gt_boxes = data[2]

            for i, id in enumerate(img_id):
                img = imdb.image_path_at(id)
                boxes = gt_boxes.numpy()[i]
                im_scale = im_info.numpy()[i][2]
                im = cv2.imread(img, cv2.IMREAD_COLOR)
                im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                                  interpolation=cv2.INTER_LINEAR)
                for i in range(len(boxes)):
                    (x1,y1,x2,y2) = boxes[i,0:4]
                    if not (np.array((x1,y1,x2,y2), dtype= np.int)== 0).all():
                        print((x1,y1,x2,y2), imdb._classes[int(boxes[i,4])])
                        im = cv2.rectangle(im, (x1,y1),(x2,y2), (0, 0, 255), 1)
                cv2.imshow(str(id), im)
                cv2.waitKey(300)
                cv2.destroyAllWindows()


        except RuntimeError as  e:
            print(e)
            #print(step,':  ',e)

            #input("wait")
            #imdb.image_path_at()
# for i in range(len(roidb)):
#     img = roidb[i]['image']
#     im = cv2.imread(img, cv2.IMREAD_COLOR)
#     boxes = roidb[i]['boxes']
#     for j in range(len(boxes)):
#         (x1, y1, x2, y2) = boxes[j, 0:4]
#         if not (np.array((x1, y1, x2, y2), dtype=np.int) == 0).all():
#             print((x1, y1, x2, y2), imdb._classes[roidb[i]['gt_classes'][j]])
#             im = cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 1)
#         cv2.imshow(str(id), im)
#         cv2.waitKey(300)
#         cv2.destroyAllWindows()
# blob = init_data(is_cuda=True)