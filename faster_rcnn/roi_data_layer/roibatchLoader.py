"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import torch
import numpy as np
import numpy.random as npr
from faster_rcnn.fast_rcnn.config import cfg
from ..roi_data_layer.minibatch import get_minibatch


class roibatchLoader(data.Dataset):
    def __init__(self, imdb, roidb, ratio_list, ratio_index, batch_size, num_classes, training=True, normalize=None):
        self._imdb = imdb
        self._roidb = roidb
        self._num_classes = num_classes
        # we make the height of image consistent to trim_height, trim_width
        self.trim_height = cfg.TRAIN.TRIM_HEIGHT
        self.trim_width = cfg.TRAIN.TRIM_WIDTH
        self.max_num_box = cfg.MAX_NUM_GT_BOXES
        self.training = training
        self.normalize = normalize
        self.ratio_list = ratio_list
        self.ratio_index = ratio_index
        self.batch_size = batch_size
        self.data_size = len(self.ratio_list)

        self.size_criteria = None

        # given the ratio_list, we want to make the ratio same for each batch.
        self.ratio_list_batch = torch.Tensor(self.data_size).zero_()
        num_batch = int(np.ceil(len(ratio_index) / batch_size))
        for i in range(num_batch):
            left_idx = i*batch_size
            right_idx = min((i+1)*batch_size-1, self.data_size-1)

            if ratio_list[right_idx] < 1:
                # for ratio < 1, we preserve the leftmost in each batch.
                target_ratio = ratio_list[left_idx]
            elif ratio_list[left_idx] > 1:
                # for ratio > 1, we preserve the rightmost in each batch.
                target_ratio = ratio_list[right_idx]
            else:
                # for ratio cross 1, we make it to be 1.
                target_ratio = 1

            self.ratio_list_batch[left_idx:(right_idx+1)] = target_ratio


    def __getitem__(self, index):

        if self.training:
            index_ratio = int(self.ratio_index[index])
        else:
            index_ratio = index

        # get the anchor index for current sample index
        # here we set the anchor index to the last one
        # sample in this group
        minibatch_db = [self._roidb[index_ratio]]
        blobs = get_minibatch(minibatch_db, self._num_classes)
        data = torch.from_numpy(blobs['data'])
        img_id = blobs['img_id']
        im_info = torch.from_numpy(blobs['im_info'])

        # we need to random shuffle the bounding box.
        data_height, data_width = data.size(1), data.size(2)
        if self.training:
            # shuffle all gt boxes info in an image
            np.random.shuffle(blobs['gt_boxes'])
            gt_boxes = torch.from_numpy(blobs['gt_boxes'])

            ########################################################
            # padding the input image to fixed size for each group #
            ########################################################

            # NOTE1: need to cope with the case where a group cover both conditions. (done)
            # NOTE2: need to consider the situation for the tail samples. (no worry)
            # NOTE3: need to implement a parallel data loader. (no worry)
            # get the index range

            # if the image need to crop, crop to the target size.
            ratio = self.ratio_list_batch[index]

            if self._roidb[index_ratio]['need_crop']:
                if ratio < 1:
                    # this means that data_width << data_height, we need to crop the
                    # data_height
                    min_y = int(torch.min(gt_boxes[:,1]))
                    max_y = int(torch.max(gt_boxes[:,3]))
                    trim_size = int(np.floor(data_width / ratio))
                    if trim_size > data_height:
                        trim_size = data_height
                    box_region = max_y - min_y + 1
                    if min_y == 0:
                        y_s = 0
                    else:
                        if (box_region-trim_size) < 0:
                            y_s_min = max(max_y-trim_size, 0)
                            y_s_max = min(min_y, data_height-trim_size)
                            if y_s_min == y_s_max:
                                y_s = y_s_min
                            else:
                                y_s = np.random.choice(range(y_s_min, y_s_max))
                        else:
                            y_s_add = int((box_region-trim_size)/2)
                            if y_s_add == 0:
                                y_s = min_y
                            else:
                                y_s = np.random.choice(range(min_y, min_y+y_s_add))
                    # crop the image
                    data = data[:, y_s:(y_s + trim_size), :, :]

                    # shift y coordiante of gt_boxes
                    gt_boxes[:, 1] = gt_boxes[:, 1] - float(y_s)
                    gt_boxes[:, 3] = gt_boxes[:, 3] - float(y_s)

                    # update gt bounding box according the trip
                    gt_boxes[:, 1].clamp_(0, trim_size - 1)
                    gt_boxes[:, 3].clamp_(0, trim_size - 1)

                else:
                    # this means that data_width >> data_height, we need to crop the
                    # data_width
                    min_x = int(torch.min(gt_boxes[:,0]))
                    max_x = int(torch.max(gt_boxes[:,2]))
                    trim_size = int(np.ceil(data_height * ratio))
                    if trim_size > data_width:
                        trim_size = data_width
                    box_region = max_x - min_x + 1
                    if min_x == 0:
                        x_s = 0
                    else:
                        if (box_region-trim_size) < 0:
                            x_s_min = max(max_x-trim_size, 0)
                            x_s_max = min(min_x, data_width-trim_size)
                            if x_s_min == x_s_max:
                                x_s = x_s_min
                            else:
                                x_s = np.random.choice(range(x_s_min, x_s_max))
                        else:
                            x_s_add = int((box_region-trim_size)/2)
                            if x_s_add == 0:
                                x_s = min_x
                            else:
                                x_s = np.random.choice(range(min_x, min_x+x_s_add))
                    # crop the image
                    data = data[:, :, x_s:(x_s + trim_size), :]

                    # shift x coordiante of gt_boxes
                    gt_boxes[:, 0] = gt_boxes[:, 0] - float(x_s)
                    gt_boxes[:, 2] = gt_boxes[:, 2] - float(x_s)
                    # update gt bounding box according the trip
                    gt_boxes[:, 0].clamp_(0, trim_size - 1)
                    gt_boxes[:, 2].clamp_(0, trim_size - 1)
            #print('before padding-------------', data.size())
            # based on the ratio, padding the image.

            if ratio < 1:
                # this means that data_width < data_height
                trim_size = int(np.floor(data_width / ratio))

                padding_data = torch.FloatTensor(int(np.ceil(data_width / ratio)),\
                                                 data_width, 3).zero_()

                padding_data[:data_height, :, :] = data[0]
                # update im_info
                im_info[0, 0] = padding_data.size(0)
                # print("height %d %d \n" %(index, anchor_idx))
            elif ratio > 1:
                # this means that data_width > data_height
                # if the image need to crop.
                padding_data = torch.FloatTensor(data_height, \
                                                 int(np.ceil(data_height * ratio)), 3).zero_()
                padding_data[:, :data_width, :] = data[0]
                im_info[0, 1] = padding_data.size(1)
            else:
                trim_size = min(data_height, data_width)
                padding_data = torch.FloatTensor(trim_size, trim_size, 3).zero_()
                padding_data = data[0][:trim_size, :trim_size, :]
                # gt_boxes.clamp_(0, trim_size)
                gt_boxes[:, :4].clamp_(0, trim_size)
                im_info[0, 0] = trim_size
                im_info[0, 1] = trim_size


            #padding_data, gt_boxes, im_info = \
            #    self.double_check_size(padding_data, gt_boxes, im_info)
            # if self.batch_size != 0:
            #     if index % self.batch_size == 0:
            #         self.size_criteria = (padding_data.size(0), padding_data.size(1))
            #     else:
            #         padding_data, gt_boxes, im_info = self.zero_padding(padding_data, gt_boxes, im_info)

            # check the bounding box:
            not_keep = (gt_boxes[:,0] == gt_boxes[:,2]) | (gt_boxes[:,1] == gt_boxes[:,3])
            keep = torch.nonzero(not_keep == 0).view(-1)

            gt_boxes_padding = torch.FloatTensor(self.max_num_box, gt_boxes.size(1)).zero_()
            if keep.numel() != 0:
                gt_boxes = gt_boxes[keep]
                num_boxes = min(gt_boxes.size(0), self.max_num_box)
                gt_boxes_padding[:num_boxes,:] = gt_boxes[:num_boxes]
            else:
                num_boxes = 0



            # permute trim_data to adapt to downstream processing
            padding_data = padding_data.permute(2, 0, 1).contiguous()
            im_info = im_info.view(3)
            return padding_data, im_info, gt_boxes_padding, num_boxes, img_id
        else:
            data = data.permute(0, 3, 1, 2).contiguous().view(3, data_height, data_width)
            im_info = im_info.view(3)

            gt_boxes = torch.FloatTensor([1,1,1,1,1])
            num_boxes = 0

            print(gt_boxes)
            return data, im_info, gt_boxes, num_boxes, img_id

    def __len__(self):
        return len(self._roidb)



    # not using functions
    def double_check_size(self, data, gt_boxes, im_info):
        check = np.array(data.size(), dtype=np.int)
        scale_list = list(cfg.TRAIN.SCALES)
        scale_list.append(cfg.TRAIN.MAX_SIZE)
        scale_list = np.unique(np.array(scale_list))
        if not np.in1d(check, scale_list).any():
            import cv2
            data = data.numpy()
            gt_boxes = gt_boxes.numpy()
            im_info = im_info.numpy()
            data = data.astype(np.float32, copy=False)
            im_shape = data.shape
            im_size_min = np.min(im_shape[0:2])
            im_size_max = np.max(im_shape[0:2])
            target_size = min(cfg.TRAIN.SCALES)
            max_size = cfg.TRAIN.MAX_SIZE
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if im_scale * im_size_max > max_size:
                im_scale = float(max_size) / float(im_size_max)
            data = cv2.resize(data, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            gt_boxes[:, 0:4] = gt_boxes[:, 0:4] * im_scale
            im_info = np.array([[data.shape[0], data.shape[1], im_info[0][2] * im_scale]], dtype=np.float32)
            data = torch.from_numpy(data)
            gt_boxes = torch.from_numpy(gt_boxes)
            im_info = torch.from_numpy(im_info)
        return data, gt_boxes, im_info

    def zero_padding(self, data, gt_boxes, im_info):
        target_height, target_width = self.size_criteria

        padding_data = torch.FloatTensor(target_height, target_width, 3).zero_()
        min_height = min(target_height, data.size(0))
        min_width = min(target_width, data.size(1))
        padding_data[:min_height,:min_width,:] = data[:min_height,:min_width,:]
        im_info = np.array([[padding_data.size(0), padding_data.size(1), im_info[0][2]]], dtype=np.float32)
        im_info = torch.from_numpy(im_info)
        gt_boxes[:, 0].clamp_(0, target_width  - 1)
        gt_boxes[:, 2].clamp_(0, target_width - 1)
        gt_boxes[:, 1].clamp_(0, target_height - 1)
        gt_boxes[:, 3].clamp_(0, target_height - 1)
        return padding_data, gt_boxes, im_info