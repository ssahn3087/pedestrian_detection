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
from faster_rcnn.roi_data_layer.sampler import sampler
from faster_rcnn.roi_data_layer.roidb import extract_roidb
from faster_rcnn.roi_data_layer.roibatchLoader import roibatchLoader
from faster_rcnn.roi_data_layer.roidb import prepare_roidb
from faster_rcnn.utils.cython_bbox import bbox_overlaps
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file




def test(model, detector, imdb, roidb):

    detector.cuda()
    detector.eval()

    print('Test Detection Performance with ', model.split('/')[-1])
    blob = init_data(is_cuda=True)
    tp, fg = 0, 0
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


def id_match_test(model, detector, imdb, roidb):
    from torch.nn.functional import cosine_similarity
    def dist(f1, f2):
        val = 0.5 * torch.sqrt(((f1 - f2) ** 2).sum(0)).data.cpu().numpy()
        return val
    def cos_sim(f1, f2):
        val = 1.0 - cosine_similarity(f1, f2, dim=0).data.cpu().numpy()
        return val

    detector.cuda()
    detector.eval()
    name_blocks = model.split('_')
    if 'euc' in name_blocks or 'log' in name_blocks:
        val_func = dist
    elif 'cls' in name_blocks:
        val_func = cos_sim
    else:
        val_func = dist

    print('Test ID Match with ', model.split('/')[-1])

    match = 0
    batch_size = imdb.num_triplet_test_images
    test_num = len(roidb)
    blob = init_data(is_cuda=True)
    num_set = int(test_num/batch_size)
    for i in range(num_set):
        features = []
        for k in range(batch_size):
            pt = batch_size * i + k
            image = cv2.imread(roidb[pt]['image'])
            gt_boxes = roidb[pt]['boxes'].astype(np.float32)
            relu = True if 'relu' in name_blocks else False
            features.append(detector.extract_feature_vector(image, blob, gt_boxes, relu=relu))
        init_val = 1e15
        for m in range(batch_size):
            for n in range(m+1, batch_size):
                val = val_func(features[m], features[n])
                if val < init_val:
                    init_val = val
                    min_m, min_n = m, n
        if roidb[batch_size * i + min_m]['ids'] == roidb[batch_size * i + min_n]['ids']:
            match += 1
        if (i+1) % 500 == 0 and i > 0: print('------------{:d}   {:.2f}%'.format(i*batch_size, match / i * 100))
    print('\tPrecision: %.2f%%, ' % (match / num_set * 100), model)
    return match / num_set * 100

def score_analysis(model, detector, imdb, roidb):
    from torch.nn.functional import cosine_similarity
    def dist(f1, f2):
        score = 0.5 * torch.sqrt(((f1 - f2) ** 2).sum(0)).data.cpu().numpy()[0]
        return score

    detector.cuda()
    detector.eval()
    name_blocks = model.split('_')
    print('Anchor-Positive, Negative Score Analysis ', model.split('/')[-1])

    pos_score = 0.
    neg_score = 0.
    bg_score = 0.
    batch_size = imdb.num_triplet_test_images
    test_num = len(roidb)
    blob = init_data(is_cuda=True)
    num_set = int(test_num/batch_size)
    for i in range(num_set):
        features = []
        bg_features = []
        for k in range(batch_size):
            pt = batch_size * i + k
            image = cv2.imread(roidb[pt]['image'])
            gt_boxes = roidb[pt]['boxes'].astype(np.float32)
            relu = True if 'relu' in name_blocks else False
            features.append(detector.extract_feature_vector(image, blob, gt_boxes, relu=relu))
            bg_features.append(detector.extract_background_features(image, blob, gt_boxes, relu=relu))

        for m in range(batch_size):
            for n in range(m+1, batch_size):
                if roidb[batch_size * i + m]['ids'] == roidb[batch_size * i + n]['ids']:
                    pos_score += dist(features[m], features[n])
                else:
                    neg_score += dist(features[m], features[n])
                    bg_score += dist(features[m], bg_features[n])
        if (i + 1) % 500 == 0 and i > 0: print(
                    '------------{:d}  pos: {:.4f} neg: {:.4f} bg: {:.4f}'\
                        .format(i * batch_size, pos_score / i, neg_score / (2*i), bg_score / (2*i)))
    pos_score /= num_set
    neg_score /= 2*num_set
    bg_score /= 2*num_set
    return pos_score, neg_score, bg_score



if __name__ == '__main__':
    # hyper-parameters
    # ------------
    imdb_name = 'CaltechPedestrians_test_triplet'
    cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
    model_dir = 'models/saved_model3/vgg16_euc'
    models = os.listdir(model_dir)
    pretrained_model = [os.path.join(model_dir, model) for model in models]
    pretrained_model.sort()
    cfg_from_file(cfg_file)
    is_resnet = cfg.RESNET.IS_TRUE
    imdb = get_imdb(imdb_name)
    prepare_roidb(imdb)
    roidb = imdb.roidb

    for model in pretrained_model:
        f = open(os.path.join(model_dir, 'precision.txt'), 'a')
        if model.endswith('txt'):
            continue
        if not is_resnet:
            detector = FasterRCNN_VGG(classes=imdb.classes, debug=False)
        else:
            detector = FasterRCNN_RES(classes=imdb.classes, debug=False)
        network.load_net(model, detector)
        match = id_match_test(model, detector, imdb, roidb) if cfg.TRIPLET.IS_TRUE else 0.
        # precision = test(model, detector, imdb, roidb)
        # pos, neg, bg = score_analysis(model, detector, imdb, roidb)
        del detector
        #f.write(model+'  -----pos: {:.4f} neg: {:.4f} bg: {:.4f}\n'.format(pos, neg, bg))
        f.write(model+'  ----{:.2f}%\n'.format(match))
        f.close()