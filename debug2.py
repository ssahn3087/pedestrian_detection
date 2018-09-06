import torch

from faster_rcnn.network import weights_normal_init
from faster_rcnn.faster_rcnn_vgg import FasterRCNN as FasterRCNN_VGG
from faster_rcnn.faster_rcnn_res import FasterRCNN as FasterRCNN_RES
from faster_rcnn.network import init_data, data_to_variable
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file
from faster_rcnn.roi_data_layer.sampler import sampler
from faster_rcnn.roi_data_layer.roidb import extract_roidb
from faster_rcnn.roi_data_layer.roibatchLoader import roibatchLoader

_DEBUG = True
imdb_name = 'CaltechPedestrians_triplet'
imdb, roidb, ratio_list, ratio_index = extract_roidb(imdb_name)
net = FasterRCNN_RES(classes=imdb.classes, debug=_DEBUG)
batch_size = 3
train_size = len(roidb)
sampler_batch = sampler(train_size, batch_size, imdb_name.split('_')[-1] == 'triplet')
dataset = roibatchLoader(imdb, roidb, ratio_list, ratio_index, batch_size,
                                                        imdb.num_classes, training=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         sampler=sampler_batch, num_workers=0)

data_iter = iter(dataloader)