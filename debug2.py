from faster_rcnn.roi_data_layer.roidb import extract_roidb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file
from collections import Counter
import cv2

cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
cfg_from_file(cfg_file)
# test_name = 'CaltechPedestrians_train'
# imdb, roidb, _, _ = extract_roidb(test_name)
imdb_name = 'coco_2017_train'

imdb, roidb, ratio_list, ratio_index = extract_roidb(imdb_name)

for db in roidb:
    path = db['image']
    im = cv2.imread(path)
    boxes = db['boxes']
    for i, det in enumerate(boxes):
        det = tuple(int(x) for x in det)
        cv2.rectangle(im, det[0:2], det[2:4], (0, 0, 255), 2)

    cv2.imshow('{}'.format(db['img_id']), im)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()