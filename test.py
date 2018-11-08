import os
import torch
import numpy as np
import cv2
import sys
import pdb
import pickle
from faster_rcnn import network
from faster_rcnn.network import init_data, data_to_variable, vis_detections
from faster_rcnn.faster_rcnn_vgg import FasterRCNN as FasterRCNN_VGG
from faster_rcnn.faster_rcnn_res import FasterRCNN as FasterRCNN_RES
from faster_rcnn.utils.timer import Timer
from faster_rcnn.roi_data_layer.roidb import extract_roidb
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file
from faster_rcnn.roi_data_layer.roibatchLoader import roibatchLoader
from faster_rcnn.fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from faster_rcnn.nms.nms_wrapper import nms

# hyper-parameters
# ------------
test_name = 'voc_2007_test'
vis = False
cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
model_dir = 'data/pretrained_model/'
output_dir = 'models/det_file/'
pre_model_name = 'voc_2007_trainval_jwyang_vgg16_0.7_b1.h5'
output_dir += pre_model_name.split('_')[-3]
pretrained_model = model_dir + pre_model_name
thresh = 0.05 if vis else 0.0
max_object = 100
rand_seed = 1024

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)

def make_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
make_dir(output_dir)

if __name__ == '__main__':
    # load data
    imdb, roidb, ratio_list, ratio_index = extract_roidb(test_name)
    imdb.competition_mode(on=True)

    print('{:d} roidb entries'.format(len(roidb)))

    t = Timer()
    file_name = pre_model_name.split('.h5')[0] + '_det.pkl'
    det_file = os.path.join(output_dir, file_name)
    try:
        if os.path.getsize(det_file) > 0:
            with open(det_file, 'rb') as fid:
                all_boxes = pickle.load(fid)
            start = t.tic()
            print('Evaluating detections')
            imdb.evaluate_detections(all_boxes, output_dir)

            end = t.tic()
            print("test time: %0.4fs" % (end - start))
    except FileNotFoundError as e:
        print(str(e))
        # start from making det file

        # load net
        is_resnet = True if 'res' in pre_model_name else False
        if is_resnet:
            model_name = cfg.RESNET.MODEL
            net = FasterRCNN_RES(classes=imdb.classes, debug=False)
            net.init_module()
        else:
            model_name = 'vgg16'
            net = FasterRCNN_VGG(classes=imdb.classes, debug=False)
            net.init_module()

        network.load_net(pretrained_model, net)
        print("load model successfully! {:s}".format(pre_model_name))

        # set net to be prepared to train
        net.cuda()
        net.eval()

        start = t.tic()
        print('det result saved in ', output_dir)

        # data prepare
        blob = init_data(is_cuda=True)
        num_images = len(imdb.image_index)
        all_boxes = [[[] for _ in range(num_images)]
                     for _ in range(imdb.num_classes)]
        dataset = roibatchLoader(imdb, roidb, ratio_list, ratio_index, 1, imdb.num_classes, training=False, normalize=False)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                 shuffle=False, num_workers=0,
                                                 pin_memory=True)
        data_iter = iter(dataloader)

        empty_array = np.transpose(np.array([[], [], [], [], []]), (1, 0))

        for i in range(num_images):
            data = next(data_iter)
            (im_data, im_info, gt_boxes, num_boxes) = data_to_variable(blob, data)
            det_tic = t.tic()
            cls_prob, bbox_pred, rois = net(im_data, im_info, gt_boxes, num_boxes)

            scores = cls_prob.data
            box_deltas = bbox_pred.data
            boxes = rois.data[:, :, 1:5]
            del cls_prob, bbox_pred, rois

            if cfg.TEST.BBOX_REG:
                if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                 + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(imdb.classes))

                pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
                pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)

            else:
                # scores.shape[1] is (cfg)BATCH_SIZE = P
                pred_boxes = np.tile(boxes, (1, scores.shape[1]))

            pred_boxes /= data[1][0][2]

            # P x n_classes
            scores = scores.squeeze()
            # P x n_classes*4
            pred_boxes = pred_boxes.squeeze()
            det_toc = t.tic()
            detect_time = det_toc - det_tic
            misc_tic = t.tic()

            if vis:
                im = cv2.imread(imdb.image_path_at(i))
                im2show = np.copy(im)
            # jth class
            for j in range(imdb.num_classes):
                inds = torch.nonzero(scores[:, j] > thresh).view(-1)
                if inds.numel() > 0:
                    cls_scores = scores[:, j][inds]
                    _, order = torch.sort(cls_scores, 0, True)
                    cls_boxes = pred_boxes[inds][:, j*4:(j+1)*4]
                    # N x 5
                    cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                    cls_dets = cls_dets[order]
                    keep = nms(cls_dets, cfg.TEST.NMS)
                    cls_dets = cls_dets[keep.view(-1).long()]
                    if vis and j != 0:
                        im2show = vis_detections(im2show, imdb.classes[j], cls_dets.cpu().numpy(), 0.3)
                    # num_class , num_images
                    all_boxes[j][i] = cls_dets.cpu().numpy()
                else:
                    all_boxes[j][i] = empty_array

            # limit to max_object detections over all classes
            if max_object > 0:
                image_scores = np.hstack([all_boxes[j][i][:, -1]
                                          for j in range(imdb.num_classes)])
                if len(image_scores) > max_object:
                    image_thresh = np.sort(image_scores)[-max_object]
                    for j in range(imdb.num_classes):
                        keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                        all_boxes[j][i] = all_boxes[j][i][keep, :]

            misc_toc = t.tic()
            nms_time = misc_toc - misc_tic

            sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                             .format(i + 1, num_images, detect_time, nms_time))
            sys.stdout.flush()
            if vis:
                cv2.imshow('test', im2show)
                cv2.waitKey(1000)
                cv2.destroyAllWindows()

        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

        print('Evaluating detections')
        imdb.evaluate_detections(all_boxes, output_dir)

        end = t.tic()
        print("test time: %0.4fs" % (end - start))