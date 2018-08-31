from faster_rcnn.roi_data_layer.roidb import extract_roidb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file
from collections import Counter


cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
cfg_from_file(cfg_file)
test_name = 'CaltechPedestrians_train'
imdb, roidb, _, _ = extract_roidb(test_name)
