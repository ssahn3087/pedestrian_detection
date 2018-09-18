import torch
import numpy as np
from faster_rcnn.network import weights_normal_init
from faster_rcnn.faster_rcnn_vgg import FasterRCNN as FasterRCNN_VGG
from faster_rcnn.faster_rcnn_res import FasterRCNN as FasterRCNN_RES
from faster_rcnn.network import init_data, data_to_variable
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file
from faster_rcnn.roi_data_layer.sampler import sampler
from faster_rcnn.roi_data_layer.roidb import extract_roidb
from faster_rcnn.roi_data_layer.roibatchLoader import roibatchLoader

_DEBUG = True
imdb_name = 'CaltechPedestrians_test_triplet'
imdb, roidb, ratio_list, ratio_index = extract_roidb(imdb_name)
net = FasterRCNN_RES(classes=imdb.classes, debug=_DEBUG)
batch_size = 3
train_size = len(roidb)
sampler_batch = sampler(train_size, batch_size, imdb_name.split('_')[-1] == 'triplet')
dataset = roibatchLoader(imdb, roidb, ratio_list, ratio_index, batch_size,
                                                        imdb.num_classes, training=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         sampler=sampler_batch, num_workers=0)
blob = init_data(is_cuda=True)
iters_per_epoch = int(train_size / batch_size)
data_iter = iter(dataloader)
from time import sleep
i = 0
for db in roidb:
    i += 1
    print(db['ids'])
    sleep(0.3)
    if i == 5 :
        i = 0
        print('-------------------------')
for step in range(iters_per_epoch):
    # get one batch
    data = next(data_iter)
    (im_data, im_info, gt_boxes, num_boxes) = data_to_variable(blob, data)
    im_info = im_info.data.cpu().numpy()
    gt_boxes = gt_boxes.data.cpu().numpy()

    boxes = np.squeeze(gt_boxes[:, 0, :])
    w_coord = np.hstack((boxes[:, 0], boxes[:, 2])).reshape(3, 2)
    h_coord = np.hstack((boxes[:, 1], boxes[:, 3])).reshape(3, 2)

    if (h_coord - im_info[:, 0].reshape(3,1) > 0).any():
        print(step)
        print('h', h_coord)
        sleep(3)
    elif (w_coord - im_info[:, 1].reshape(3,1) > 0).any():
        print(step)
        print('w', w_coord)
        sleep(3)

    if step % 300 == 0:
        print(step)