import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from faster_rcnn.utils.blob import im_list_to_blob
from faster_rcnn.nms.nms_wrapper import nms
from faster_rcnn.rpn_msr.proposal_layer import proposal_layer as proposal_layer_py
from faster_rcnn.rpn_msr.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from faster_rcnn.rpn_msr.proposal_target_layer import proposal_target_layer as proposal_target_layer_py
from faster_rcnn.fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes

from faster_rcnn.network import weights_normal_init
from faster_rcnn.network import _smooth_l1_loss
from faster_rcnn.network import Conv2d, FC
from faster_rcnn.network import get_triplet_rois
# from roi_pooling.modules.roi_pool_py import RoIPool
from faster_rcnn.roi_align.modules.roi_align import RoIAlign
from faster_rcnn.roi_pooling.modules.roi_pool import RoIPool
from faster_rcnn.vgg16 import VGG16
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file

cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
cfg_from_file(cfg_file)


def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
    dets = torch.cat((pred_boxes, scores.unsqueeze(1)), 1)
    keep = nms(dets, nms_thresh).long().view(-1)
    if inds is None:
        return pred_boxes[keep], scores[keep]
    return pred_boxes[keep], scores[keep], inds[keep]


class RPN(nn.Module):

    def __init__(self, debug=False):
        super(RPN, self).__init__()

        self.anchor_scales = cfg.ANCHOR_SCALES
        self.anchor_ratios = cfg.ANCHOR_RATIOS
        self.feat_stride = cfg.FEAT_STRIDE[0]

        self.features = VGG16(bn=False)
        self.conv1 = Conv2d(512, 512, 3, same_padding=True)
        self.score_conv = Conv2d(512, len(self.anchor_scales) * len(self.anchor_ratios) * 2, 1, relu=False,
                                 same_padding=False)
        self.bbox_conv = Conv2d(512, len(self.anchor_scales) * len(self.anchor_ratios) * 4, 1, relu=False,
                                same_padding=False)

        # define proposal layer
        self.proposal_layer = proposal_layer_py(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # define anchor target layer
        self.anchor_target_layer = anchor_target_layer_py(self.feat_stride, self.anchor_scales, self.anchor_ratios)

        # loss
        self.cross_entropy = 0
        self.loss_box = 0

        # for log
        self.debug = debug

    @property
    def loss(self):
        return self.cross_entropy + self.loss_box

    def forward(self, im_data, im_info, gt_boxes, num_boxes):

        features = self.features(im_data)
        batch_size = features.size(0)
        rpn_conv1 = self.conv1(features)
        # rpn score
        rpn_cls_score = self.score_conv(rpn_conv1)

        rpn_cls_score_reshape = self.reshape_layer(rpn_cls_score, 2)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, dim=1)
        rpn_cls_prob = self.reshape_layer(rpn_cls_prob_reshape, len(self.anchor_scales) * len(self.anchor_ratios) * 2)

        # rpn boxes
        rpn_bbox_pred = self.bbox_conv(rpn_conv1)
        # proposal layer
        cfg_key = 'TRAIN' if self.training else 'TEST'
        rois = self.proposal_layer((rpn_cls_prob.data, rpn_bbox_pred.data, im_info,
                                    cfg_key))

        self.cross_entropy = 0
        self.loss_box = 0

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None
            rpn_data = self.anchor_target_layer((rpn_cls_score.data, gt_boxes, im_info, num_boxes))
            self.cross_entropy, self.loss_box = self.build_loss(rpn_cls_score_reshape, rpn_bbox_pred,
                                                                rpn_data, batch_size)

        return features, rois

    def build_loss(self, rpn_cls_score_reshape, rpn_bbox_pred, rpn_data, batch_size):

        # compute classification loss
        rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
        rpn_label = rpn_data[0].view(batch_size, -1)

        rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))
        rpn_cls_score = torch.index_select(rpn_cls_score.view(-1, 2), 0, rpn_keep)
        rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
        rpn_label = Variable(rpn_label.long())

        ce_weights = torch.ones(rpn_cls_score.size(1))
        fg_box = torch.sum(rpn_label.data.ne(0))
        bg_box = rpn_label.data.numel() - fg_box
        if bg_box > 0:
            ce_weights[0] = float(fg_box) / bg_box
        ce_weights = ce_weights.cuda()
        self.cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label, weight=ce_weights).mean()
        if self.debug:
            maxv, predict = rpn_cls_score.data.max(1)
            tp = rpn_label.data.eq(predict) * rpn_label.data.ne(0)
            self.tp = torch.sum(tp) if fg_box > 0 else 0
            self.fg_box = fg_box

        rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]

        # compute bbox regression loss
        rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
        rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
        rpn_bbox_targets = Variable(rpn_bbox_targets)

        self.loss_box = _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                        rpn_bbox_outside_weights, sigma=3, dim=[1, 2, 3]).mean()

        return self.cross_entropy, self.loss_box

    @staticmethod
    def reshape_layer(x, d):
        input_shape = x.size()
        # x = x.permute(0, 3, 1, 2)
        # b c w h
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        # x = x.permute(0, 2, 3, 1)
        return x

    def load_from_npz(self, params):
        # params = np.load(npz_file)
        self.features.load_from_npz(params)

        pairs = {'conv1.conv': 'rpn_conv/3x3', 'score_conv.conv': 'rpn_cls_score', 'bbox_conv.conv': 'rpn_bbox_pred'}
        own_dict = self.state_dict()
        for k, v in pairs.items():
            key = '{}.weight'.format(k)
            param = torch.from_numpy(params['{}/weights:0'.format(v)]).permute(3, 2, 0, 1)
            own_dict[key].copy_(param)

            key = '{}.bias'.format(k)
            param = torch.from_numpy(params['{}/biases:0'.format(v)])
            own_dict[key].copy_(param)


class FasterRCNN(nn.Module):
    n_classes = 21
    # for pacal_voc but flexible
    classes = np.asarray(['__background__',
                          'aeroplane', 'bicycle', 'bird', 'boat',
                          'bottle', 'bus', 'car', 'cat', 'chair',
                          'cow', 'diningtable', 'dog', 'horse',
                          'motorbike', 'person', 'pottedplant',
                          'sheep', 'sofa', 'train', 'tvmonitor'])
    PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
    SCALES = cfg.TRAIN.SCALES
    MAX_SIZE = cfg.TRAIN.MAX_SIZE

    def __init__(self, classes=None, debug=False):
        super(FasterRCNN, self).__init__()

        if classes is not None:
            self.classes = np.asarray(classes)
            self.n_classes = len(classes)

        self.rpn = RPN(debug=debug)
        self.proposal_target_layer = proposal_target_layer_py(self.n_classes)
        if cfg.POOLING_MODE == 'align':
            self.roi_pool = RoIAlign(7, 7, 1.0 / 16)
        elif cfg.POOLING_MODE == 'pool':
            self.roi_pool = RoIPool(7, 7, 1.0 / 16)
        self.fc6 = FC(512 * 7 * 7, 4096)
        self.fc7 = FC(4096, 4096)
        self.score_fc = FC(4096, self.n_classes, relu=False)
        self.bbox_fc = FC(4096, self.n_classes * 4, relu=False)
        self.fc_sim = FC(512 * 7 * 7, 4096, relu=False)

        # loss
        self.cross_entropy = 0
        self.loss_box = 0
        self.triplet_loss = 0
        # for log
        self.debug = debug
        if cfg.TRIPLET.IS_TRUE:
            if self.debug:
                self.set = 0
                self.match = 0
            if cfg.TRIPLET.LOSS == 'euc':
                self.loss_triplet = self.euclidean_distance_loss
            elif cfg.TRIPLET.LOSS == 'log':
                self.loss_triplet = self.lossless_triplet_loss
            elif cfg.TRIPLET.LOSS == 'cls':
                self.loss_triplet = self.cross_entropy_cosine_sim
        self.init_module = self._init_faster_rcnn_vgg16
    @property
    def loss(self):
        return self.cross_entropy + self.loss_box + self.triplet_loss

    def forward(self, im_data, im_info, gt_boxes, num_boxes):
        batch_size = im_data.size(0)

        im_info = im_info.data
        gt_boxes = gt_boxes.data
        num_boxes = num_boxes.data
        features, rois = self.rpn(im_data, im_info, gt_boxes, num_boxes)

        if self.training:
            roi_data = self.proposal_target_layer(rois, gt_boxes, num_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
            rois_label = Variable(rois_label.view(-1).long())
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            self.cross_entropy = 0
            self.loss_box = 0
            self.triplet_loss = 0

        rois = Variable(rois)

        # roi pool
        pooled_features = self.roi_pool(features, rois.view(-1, 5))

        x = pooled_features.view(pooled_features.size()[0], -1)
        x = self.fc6(x)
        x = F.dropout(x, training=self.training)
        x = self.fc7(x)
        x = F.dropout(x, training=self.training)

        cls_score = self.score_fc(x)
        cls_prob = F.softmax(cls_score, dim=1)
        bbox_pred = self.bbox_fc(x)

        self.cross_entropy = 0
        self.loss_box = 0
        self.triplet_loss = 0

        if self.training:
            # select the corresponding columns according to roi labels
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            bbox_pred_select = torch.gather(bbox_pred_view, 1,
                                            rois_label.view(rois_label.size(0), 1, 1).expand(rois_label.size(0), 1, 4))
            bbox_pred = bbox_pred_select.squeeze(1)
            self.cross_entropy, self.loss_box = self.build_loss(cls_score, bbox_pred, roi_data)
            # triplet loss
            if cfg.TRIPLET.IS_TRUE:
                triplet_rois = get_triplet_rois(rois, rois_label, cfg.TRIPLET.MAX_BG)
                triplet_features = self.roi_pool(features, triplet_rois.view(-1, 5))
                triplet_features = triplet_features.view(triplet_features.size(0), -1)
                triplet_features = self.fc_sim(triplet_features)
                self.triplet_loss = self.loss_triplet(triplet_features)

        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        return cls_prob, bbox_pred, rois

    def build_loss(self, cls_score, bbox_pred, roi_data):
        # classification loss
        rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data
        rois_label = Variable(rois_label.view(-1).long())
        rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
        rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
        rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))

        label = rois_label
        fg_cnt = torch.sum(label.data.ne(0))
        bg_cnt = label.data.numel() - fg_cnt

        # for log
        if self.debug:
            maxv, predict = cls_score.data.max(1)
            tp = label.data.eq(predict) * label.data.ne(0)
            tf = label.data.eq(0) * predict.eq(0)
            self.tp = torch.sum(tp) if fg_cnt > 0 else 0
            self.tf = torch.sum(tf)
            self.fg_cnt = fg_cnt
            self.bg_cnt = bg_cnt

        ce_weights = torch.ones(cls_score.size(1))
        if bg_cnt > 0:
            ce_weights[0] = float(fg_cnt) / bg_cnt
        ce_weights = ce_weights.cuda()
        # cross_entropy = F.cross_entropy(cls_score, rois_label).mean()
        cross_entropy = F.cross_entropy(cls_score, rois_label, weight=ce_weights).mean()
        loss_box = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws).mean()

        return cross_entropy, loss_box

    def euclidean_distance_loss(self, triplet_features):
        """
        euclidean_distance_loss
        """
        from torch.nn.functional import normalize
        anchor = normalize(triplet_features[0].view(-1), dim=0)
        positive = normalize(triplet_features[1].view(-1), dim=0)
        negative = normalize(triplet_features[2].view(-1), dim=0)

        pos_dist = ((anchor - positive) ** 2).sum(0)
        neg_dist = ((anchor - negative) ** 2).sum(0)

        if triplet_features.size(0) > 3:
            rem_features = normalize(triplet_features[3:], dim=1)
            bg_size = rem_features.size(0)
            rem_features = rem_features.view(bg_size, -1)
            rem_dist = ((anchor - rem_features) ** 2).sum(1).sum(0) / float(bg_size)
        else:
            rem_dist = 0.
        if self.debug:
            self.set += 1
            self.match += 1 if pos_dist.data[0] < neg_dist.data[0] and pos_dist.data[0] < rem_dist.data[0] else 0
        loss = pos_dist + (1 - 0.2 * (neg_dist + rem_dist)).clamp(min=0.)
        return loss

    def lossless_triplet_loss(self, triplet_features):
        """
        euclidean_distance_loss
        """
        from math import isnan
        from torch.nn.functional import normalize
        anchor = normalize(triplet_features[0].view(-1), dim=0)
        positive = normalize(triplet_features[1].view(-1), dim=0)
        negative = normalize(triplet_features[2].view(-1), dim=0)

        pos_dist = 0.5 * torch.sqrt(((anchor - positive) ** 2).sum(0))
        neg_dist = 0.5 * torch.sqrt(((anchor - negative) ** 2).sum(0))

        if triplet_features.size(0) > 3:
            rem_features = normalize(triplet_features[3:], dim=1)
            bg_size = rem_features.size(0)
            rem_features = rem_features.view(bg_size, -1)
            rem_dist = 0.5 * torch.sqrt(((anchor - rem_features) ** 2).sum(1)).sum(0) / float(bg_size)
        else:
            rem_dist = 0.
        if self.debug:
            self.set += 1
            self.match += 1 if pos_dist.data[0] < neg_dist.data[0] and pos_dist.data[0] < rem_dist.data[0] else 0
        epsilon = 1e-12
        loss = - torch.log(-pos_dist + 1.0 + epsilon) - torch.log(neg_dist + epsilon) - torch.log(rem_dist + epsilon)
        loss = 0. if isnan(loss.data[0]) else loss
        return loss

    def cross_entropy_cosine_sim(self, triplet_features):
        """
        euclidean_distance_loss
        """
        from torch.nn.functional import normalize, cosine_similarity
        anchor = normalize(triplet_features[0].view(-1), dim=0)
        positive = normalize(triplet_features[1].view(-1), dim=0)
        negative = normalize(triplet_features[2].view(-1), dim=0)
        scores = Variable(torch.zeros((2, 2)).cuda())
        scores[:, 0].data.copy_(cosine_similarity(anchor, positive, dim=0))
        scores[0, 1] = 1.0 - cosine_similarity(anchor, negative, dim=0)
        labels = Variable(torch.zeros(2).long().cuda())
        labels[1] = 1
        if triplet_features.size(0) > 3:
            rem_features = normalize(triplet_features[3:], dim=1)
            bg_size = rem_features.size(0)
            anchor = anchor.unsqueeze(0)
            rem_features = rem_features.view(bg_size, -1)
            scores[1, 1] = (1.0 - cosine_similarity(rem_features, anchor, dim=1)).sum(0) / float(bg_size)
        else:
            scores = scores[:1]
            labels = labels[:1]
        loss = F.cross_entropy(scores, labels).sum()
        return loss
    def reset_match_count(self):
        self.match = 0
        self.set = 0

    def interpret_faster_rcnn(self, cls_prob, bbox_pred, rois, im_info, nms=True, min_score=0.0, nms_thresh=0.3):
        scores = cls_prob.data.squeeze()
        # find class
        scores, inds = scores.max(1)
        keep = ((inds > 0) & (scores >= min_score)).nonzero().squeeze()
        scores, inds = scores[keep], inds[keep]

        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data.squeeze()[keep]
        boxes = rois.data.squeeze()[:, 1:5][keep]

        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                         + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()

        box_deltas = box_deltas.view(-1, 4 * self.n_classes)
        box_deltas = torch.cat([box_deltas[i, (inds[i] * 4): (inds[i] * 4 + 4)] \
                                for i in range(len(inds))], 0)
        box_deltas = box_deltas.view(-1, 4)
        boxes, box_deltas = boxes.unsqueeze(0), box_deltas.unsqueeze(0)
        pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
        pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        pred_boxes = pred_boxes.squeeze()
        pred_boxes /= im_info.data[0][2]
        # nms
        if nms and pred_boxes.size(0) > 0:
            pred_boxes, scores, inds = nms_detections(pred_boxes, scores, nms_thresh, inds=inds)
        pred_boxes = pred_boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        inds = inds.cpu().numpy()

        return pred_boxes, scores, self.classes[inds]


    def detect(self, image, blob, thr=0.3, nms_thresh=0.3):
        im_data, im_scales = self.get_image_blob(image)
        im_info = np.array(
            [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
            dtype=np.float32)

        im_data_pt = torch.from_numpy(im_data)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info)
        (im_data, im_info, gt_boxes, num_boxes) = blob
        im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.data.resize_(1, 1, 5).zero_()
        num_boxes.data.resize_(1).zero_()

        cls_prob, bbox_pred, rois = self(im_data, im_info, gt_boxes, num_boxes)
        pred_boxes, scores, classes = \
            self.interpret_faster_rcnn(cls_prob, bbox_pred, rois, im_info, image.shape, \
                                       min_score=thr, nms_thresh=nms_thresh)
        return pred_boxes, scores, classes

    def extract_feature_vector(self, image, blob, gt_boxes, norm=False):
        from torch.nn.functional import normalize
        im_data, im_scales = self.get_image_blob(image)
        im_info = np.array(
            [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
            dtype=np.float32)
        gt_boxes *= im_scales[0]
        gt_boxes = np.hstack((gt_boxes, np.array([[1.]])))
        gt_boxes = gt_boxes[np.newaxis, :]
        im_data_pt = torch.from_numpy(im_data)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info)
        gt_boxes_pt = torch.from_numpy(gt_boxes)
        (im_data, im_info, gt_boxes, num_boxes) = blob
        im_data.data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.data.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.data.resize_(gt_boxes_pt.size()).copy_(gt_boxes_pt)
        num_boxes.data.resize_(1).fill_(1)
        assert im_data.size(0) == 1

        features, rois = self.rpn(im_data, im_info.data, gt_boxes.data, num_boxes.data)
        # rois, rois_label, _, _, _ = self.proposal_target_layer(rois, gt_boxes.data, num_boxes.data)
        # rois_label = Variable(rois_label.view(-1).long())
        # triplet_rois = Variable(torch.zeros(1, rois.size(2))).cuda()
        #
        # indice = torch.nonzero(rois_label.data == 1)
        triplet_rois = Variable(torch.zeros(1, rois.size(2))).cuda()
        triplet_rois[0, 1:5] = gt_boxes.data[0, 0, :4]
        triplet_features = self.roi_pool(features, triplet_rois.view(-1, 5))
        triplet_features = self.fc7(self.fc6(triplet_features.view(triplet_features.size(0), -1))).squeeze()
        triplet_features = normalize(triplet_features, dim=0) if norm else triplet_features
        return triplet_features

    def get_image_blob_noscale(self, im):
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= self.PIXEL_MEANS

        processed_ims = [im]
        im_scale_factors = [1.0]

        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)


    def get_image_blob(self, im):
        """Converts an image into a network input.
           Arguments:
               im (ndarray): a color image in BGR order
           Returns:
               blob (ndarray): a data blob holding an image pyramid
               im_scale_factors (list): list of image scales (relative to im) used
                   in the image pyramid
           """
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= self.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in self.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > self.MAX_SIZE:
                im_scale = float(self.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)


    def load_pretrained_vgg16(self, fname):
        import os
        assert os.path.exists(fname), \
            'Pretrained model does not exist: {}'.format(fname)
        params = np.load(fname, encoding='latin1').item()
        print("loaded vgg16 model from %s" % fname)
        # vgg16
        vgg16_dict = self.rpn.features.state_dict()
        for name, val in vgg16_dict.items():
            # # print name
            # # print val.size()
            # # print param.size()
            if name.find('bn.') >= 0:
                continue
            i, j = int(name[4]), int(name[6]) + 1
            ptype = 'weights' if name[-1] == 't' else 'biases'
            key = 'conv{}_{}'.format(i, j)
            param = torch.from_numpy(params[key][ptype])

            if ptype == 'weights':
                param = param.permute(3, 2, 0, 1)

            val.copy_(param)

        # fc6 fc7
        frcnn_dict = self.state_dict()
        pairs = {'fc6.fc': 'fc6', 'fc7.fc': 'fc7'}
        for k, v in pairs.items():
            key = '{}.weight'.format(k)
            param = torch.from_numpy(params[v]['weights']).permute(1, 0)
            frcnn_dict[key].copy_(param)

            key = '{}.bias'.format(k)
            param = torch.from_numpy(params[v]['biases'])
            frcnn_dict[key].copy_(param)


    def _init_faster_rcnn_vgg16(self):
        weights_normal_init(self)
        self.load_pretrained_vgg16(self.rpn.features.model_path)