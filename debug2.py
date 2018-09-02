from faster_rcnn.network import weights_normal_init
from faster_rcnn.faster_rcnn_vgg import FasterRCNN as FasterRCNN_VGG
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file
from faster_rcnn.roi_data_layer.roidb import extract_roidb

_DEBUG = True
imdb_name = 'voc_2007_trainval'
imdb, roidb, ratio_list, ratio_index = extract_roidb(imdb_name)
net = FasterRCNN_VGG(classes=imdb.classes, debug=_DEBUG)

cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
cfg_from_file(cfg_file)