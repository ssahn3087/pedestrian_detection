import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import numpy.random as npr

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        if k in h5f.keys():
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)
        elif k.startswith('fc_sim'):
            k = k.replace('fc_sim', 'fc6')
            param = torch.from_numpy(np.asarray(h5f[k]))
            v.copy_(param)
        else:
            continue


def load_net_pedestrians(fname, net, person_key):
    import h5py
    h5f = h5py.File(fname, mode='r')
    cls_related_part = list(net.state_dict().keys())[-4:]
    own_dict = net.state_dict()
    need_index = [0, person_key]
    # num_classses = size of score_fc.fc.bias in net
    num_classes = len(list(net.state_dict().values())[-3])

    irrelvant_indices = np.where(np.isin(np.arange(21), need_index) \
                                 == False)[0]
    for k, v in own_dict.items():
        data = np.asarray(h5f[k])
        if k in cls_related_part:
            if str(k).startswith('score'):
                if num_classes == 2:
                    data = np.delete(data, irrelvant_indices, axis=0)
                else:
                    data[irrelvant_indices] = 0.
            elif str(k).startswith('bbox'):
                data = data.reshape(21, 4, -1)
                if num_classes == 2:
                    data = np.delete(data, irrelvant_indices, axis=0)
                else:
                    data[irrelvant_indices] = 0.
                data = data.reshape(num_classes * 4, -1)
        param = torch.from_numpy(data)
        v.copy_(param)


def load_pretrained_npy(fname, faster_rcnn_model):
    params = np.load(fname, encoding='latin1').item()
    # vgg16
    vgg16_dict = faster_rcnn_model.rpn.features.state_dict()
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
    frcnn_dict = faster_rcnn_model.state_dict()
    pairs = {'fc6.fc': 'fc6', 'fc7.fc': 'fc7'}
    for k, v in pairs.items():
        key = '{}.weight'.format(k)
        param = torch.from_numpy(params[v]['weights']).permute(1, 0)
        frcnn_dict[key].copy_(param)

        key = '{}.bias'.format(k)
        param = torch.from_numpy(params[v]['biases'])
        frcnn_dict[key].copy_(param)


def _smooth_l1_loss(bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights, sigma=1.0, dim=[1]):
    sigma_2 = sigma ** 2
    box_diff = bbox_pred - bbox_targets
    in_box_diff = bbox_inside_weights * box_diff
    abs_in_box_diff = torch.abs(in_box_diff)
    smoothL1_sign = (abs_in_box_diff < 1. / sigma_2).detach().float()
    in_loss_box = torch.pow(in_box_diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
                  + (abs_in_box_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)
    out_loss_box = bbox_outside_weights * in_loss_box
    loss_box = out_loss_box
    for i in sorted(dim, reverse=True):
        loss_box = loss_box.sum(i)
    loss_box = loss_box.mean()

    return loss_box


def init_data(is_cuda=True):

    im_data = Variable(torch.FloatTensor(1).cuda()) if is_cuda else Variable(torch.FloatTensor(1))
    im_info = Variable(torch.FloatTensor(1).cuda()) if is_cuda else Variable(torch.FloatTensor(1))
    gt_boxes = Variable(torch.FloatTensor(1).cuda()) if is_cuda else Variable(torch.FloatTensor(1))
    num_boxes = Variable(torch.LongTensor(1).cuda()) if is_cuda else Variable(torch.LongTensor(1))

    return (im_data,im_info,gt_boxes,num_boxes)


def data_to_variable(blob, data):
    blob[0].data.resize_(data[0].size()).copy_(data[0])
    blob[1].data.resize_(data[1].size()).copy_(data[1])
    blob[2].data.resize_(data[2].size()).copy_(data[2])
    blob[3].data.resize_(data[3].size()).copy_(data[3])
    return blob


def np_to_variable(x, is_cuda=True, dtype=torch.FloatTensor):
    v = Variable(torch.from_numpy(x).type(dtype))
    if is_cuda:
        v = v.cuda()
    return v


def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm ** 2
    totalnorm = np.sqrt(totalnorm)

    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            p.grad.mul_(norm)

def train_net_params(net, cfg, lr):
    params = []
    for key, value in dict(net.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                params += [{'params': [value], 'lr': lr * (cfg.TRAIN.DOUBLE_BIAS + 1), \
                            'weight_decay': lr and cfg.TRAIN.WEIGHT_DECAY or 0}]
            else:
                params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
    return params


def get_triplet_rois(rois, rois_label, bg_num=0):

    # first and second batch are pos
    # third batch is neg
    rois = rois.data
    batch_size = rois.size(0)
    assert batch_size == 3, 'triplet loss must be based on batch size (3)'
    rois_label = rois_label.view(batch_size, -1)
    triplet_rois = Variable(torch.zeros(3 + bg_num, rois.size(2))).cuda()
    for i in range(batch_size):
        indices = torch.nonzero(rois_label[i].data == 1)
        rand_id = torch.from_numpy(npr.choice(indices.size(0), 1)).long().cuda()
        triplet_rois[i, :] = rois[i][rand_id]
    if bg_num != 0:
        indices = torch.nonzero(rois_label[2].data == 0)
        if indices.nelement() == 0:
            return triplet_rois[:3]
        rand_id = torch.from_numpy(npr.choice(indices.size(0), bg_num)).long().cuda()
        triplet_rois[3:, :] = rois[2][rand_id]
    return triplet_rois
