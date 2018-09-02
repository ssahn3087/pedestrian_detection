import os
import torch
import numpy as np
from torch.autograd import Variable
import cv2
from faster_rcnn import network
from faster_rcnn.network import init_data, data_to_variable
from faster_rcnn.network import train_net_params
from faster_rcnn.faster_rcnn_vgg import FasterRCNN as FasterRCNN_VGG
from faster_rcnn.faster_rcnn_res import FasterRCNN as FasterRCNN_RES
from faster_rcnn.datasets.factory import get_imdb
from faster_rcnn.roi_data_layer.roidb import prepare_roidb
from faster_rcnn.utils.cython_bbox import bbox_overlaps
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file




def test(model, detector, imdb, roidb):

    detector.cuda()
    detector.eval()

    print('load model successfully!')
    blob = init_data(is_cuda=True)
    tp, fg = 0, 0
    print('Test Phase with ', model)
    test_num = len(roidb)
    # display_interval = 1000
    for i in range(test_num):
        gt_boxes = roidb[i]['boxes']
        gt_classes = roidb[i]['gt_classes']
        fg += int(len(gt_boxes))
        image = cv2.imread(roidb[i]['image'])
        try:
            dets, scores, classes = detector.detect(image, blob, thr=0.7, nms_thresh=0.3)
            # dets : N x 4, gt_boxes : K x 4
            # overlaps : N x K overlaps score
            overlaps = bbox_overlaps(np.ascontiguousarray(dets, dtype=np.float) \
                                     , np.ascontiguousarray(gt_boxes, dtype=np.float))
            # max : K max overlaps score about N dets
            overlaps = np.multiply(overlaps, overlaps > cfg.TEST.RPN_NMS_THRESH)
            candidates = overlaps.argmax(axis=0)
            for i, arg in enumerate(candidates):
                if imdb._class_to_ind[classes[arg]] == gt_classes[i]:
                    tp += 1
        except:
            pass
        # if (i % display_interval == 0) and i > 0:
        #     print('\t---{}  Precision: {:.2f}%, '.format(i, (tp / fg * 100)), model)
    print('\tPrecision: %.2f%%, ' % (tp/fg*100), model)
    return tp/fg*100

if __name__ == '__main__':
    # hyper-parameters
    # ------------
    imdb_name = 'CaltechPedestrians_test'
    cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
    model_dir = 'data/test_phase/'
    models = os.listdir(model_dir)
    pretrained_model = [os.path.join(model_dir, model) for model in models]
    pretrained_model.sort()
    cfg_from_file(cfg_file)
    is_resnet = cfg.RESNET.IS_TRUE
    imdb = get_imdb(imdb_name)
    prepare_roidb(imdb)
    roidb = imdb.roidb
    f = open(os.path.join(model_dir,'precision.txt'), 'w')

    for model in pretrained_model:
        if model.endswith('txt'):
            continue
        if not is_resnet:
            detector = FasterRCNN_VGG(classes=imdb.classes, debug=False)
        else:
            detector = FasterRCNN_RES(classes=imdb.classes, debug=False)
        network.load_net(model, detector)
        precision = test(model, detector, imdb, roidb)
        f.write(model+'  ----{:.2f}%\n'.format(precision))
    f.close()