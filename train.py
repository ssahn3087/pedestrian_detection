import os
import torch
import numpy as np
import math
from torch.autograd import Variable
from datetime import datetime
from faster_rcnn import network
from faster_rcnn.network import init_data, data_to_variable
from faster_rcnn.network import train_net_params
from faster_rcnn.faster_rcnn_vgg import FasterRCNN as FasterRCNN_VGG
from faster_rcnn.faster_rcnn_res import FasterRCNN as FasterRCNN_RES
from faster_rcnn.utils.timer import Timer
from test_model import test, id_match_test
from faster_rcnn.roi_data_layer.sampler import sampler
from faster_rcnn.roi_data_layer.roidb import extract_roidb
from faster_rcnn.roi_data_layer.roibatchLoader import roibatchLoader
from faster_rcnn.fast_rcnn.config import cfg, cfg_from_file

try:
    from termcolor import cprint
except ImportError:
    cprint = None

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


def log_print(text, color='blue', on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)



# hyper-parameters
# ------------

imdb_name = 'coco_2017_train'
test_name = 'coco_2017_val'
#imdb_name = 'CaltechPedestrians_train_triplet'
#test_name = 'CaltechPedestrians_test_triplet'


cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
model_dir = 'data/pretrained_model/'
output_dir = 'models/saved_model3'
pre_model_name = 'CaltechPedestrians_train_1_vgg16_0.7_b1.h5'
pretrained_model = model_dir + pre_model_name


start_epoch = 1

end_epoch = 10
lr_decay_step = 3
lr_decay = 0.8
rand_seed = 1024


_DEBUG = True
use_tensorboard = True
remove_all_log = False # remove all historical experiments in TensorBoard
exp_name = None  # the previous experiment name in TensorBoard

# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)
fg_thresh = cfg.TRAIN.RPN_POSITIVE_OVERLAP
is_resnet = cfg.RESNET.IS_TRUE
batch_size = cfg.TRAIN.IMS_PER_BATCH
lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval = cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS
save_interval = cfg.TRAIN.SNAPSHOT_ITERS
# load data
imdb, roidb, ratio_list, ratio_index = extract_roidb(imdb_name)
test_imdb, test_roidb, _, _ = extract_roidb(test_name)
train_size = len(roidb)
sampler_batch = sampler(train_size, batch_size, cfg.TRIPLET.IS_TRUE)
dataset = roibatchLoader(imdb, roidb, ratio_list, ratio_index, batch_size,
                                                        imdb.num_classes, training=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         sampler=sampler_batch, num_workers=0)


# load net
if is_resnet:
    model_name = cfg.RESNET.MODEL
    net = FasterRCNN_RES(classes=imdb.classes, debug=_DEBUG)
    net.init_module()
else:
    model_name = 'vgg16'
    net = FasterRCNN_VGG(classes=imdb.classes, debug=_DEBUG)
    net.init_module()
if cfg.TRIPLET.IS_TRUE:
    model_name += '_' + cfg.TRIPLET.LOSS
#network.load_net(pretrained_model, net)
#network.load_net_pedestrians(pretrained_model, net, 1)

blob = init_data(is_cuda=True)

# set net to be prepared to train
net.cuda()
params = train_net_params(net, cfg, lr)
optimizer = torch.optim.SGD(params, momentum=momentum)


def make_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
make_dir(output_dir)

# tensorboad
use_tensorboard = use_tensorboard and CrayonClient is not None
if use_tensorboard:
    print('TENSORBOARD IS ON')
    cc = CrayonClient(hostname='127.0.0.1')

    if remove_all_log:
        cc.remove_all_experiments()
    if exp_name is None:
        name = '{}_{}'.format(imdb_name, model_name)
        exp_name = datetime.now().strftime(name+'_%m-%d_%H-%M')
        exp = cc.create_experiment(exp_name)
    else:
        exp = cc.open_experiment(exp_name)

iters_per_epoch = int(train_size / batch_size)
# training
train_loss = 0
previous_precision = 0.
descend = 0
step_cnt = 0
cnt = 0
re_cnt = False

t = Timer()
t.tic()
from math import isnan
for epoch in range(start_epoch, end_epoch+1):
    pf, tot = 0., 0
    tp, tf, fg, bg, tp_box, fg_box = 0., 0., 0, 0, 0., 0
    rpn_cls, rpn_box, rcnn_cls, rcnn_box, sim_loss = 0., 0., 0., 0., 0.
    net.train()
    if epoch > 1 and (epoch-1) % lr_decay_step == 0:
        lr *= lr_decay
        params = train_net_params(net, cfg, lr)
        optimizer = torch.optim.SGD(params, momentum=momentum)

    data_iter = iter(dataloader)
    for step in range(iters_per_epoch):

        # get one batch
        data = next(data_iter)
        (im_data, im_info, gt_boxes, num_boxes) = data_to_variable(blob, data)
        # forward
        net.zero_grad()
        net(im_data, im_info, gt_boxes, num_boxes)

        if _DEBUG:
            tp += float(net.tp)
            tf += float(net.tf)
            fg += net.fg_cnt
            bg += net.bg_cnt
            tp_box += float(net.rpn.tp)
            fg_box += net.rpn.fg_box
            rpn_box += net.rpn.cross_entropy.data.cpu().numpy()[0]
            rpn_cls += net.rpn.loss_box.data.cpu().numpy()[0]
            rcnn_box += net.loss_box.data.cpu().numpy()[0]
            rcnn_cls += net.cross_entropy.data.cpu().numpy()[0]
            sim_loss += net.triplet_loss.data.cpu().numpy()[0] if cfg.TRIPLET.IS_TRUE else 0.
        loss = net.rpn.loss + net.loss
        if isnan(loss):
            print(gt_boxes)
            print(net.rpn.loss, net.loss)
        train_loss += loss.data[0]
        step_cnt += 1
        cnt += 1

        # backward
        optimizer.zero_grad() # clear grad
        loss.backward()
        network.clip_gradient(net, 10.)
        optimizer.step()

        if step % disp_interval == 0:
            duration = t.toc(average=False)
            fps = step_cnt / duration

            log_text = 'step %d, loss: %.4f, fps: %.2f (%.2fs per batch) --[epoch %2d] --[iter %4d/%4d]' % (
                step, train_loss / step_cnt, fps, 1./fps, epoch, step, iters_per_epoch)
            log_print(log_text, color='green', attrs=['bold'])

            if _DEBUG:
                if fg == 0 or bg == 0:
                    pass
                else:
                    tot += 1
                    pf += tp/fg*100
                    match_rate = net.match/net.set * 100. if cfg.TRIPLET.IS_TRUE else 0.
                    log_print('\tEP: %.2f%%, TP: %.2f%%, TF: %.2f%%, fg/bg=(%d/%d), TD: %.2f%%' %
                        (tp_box/fg_box*100, tp/fg*100., tf/bg*100., fg/step_cnt, bg/step_cnt, match_rate))
                    log_print('\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f, sim_loss: %.4f' % (
                        rpn_cls/step_cnt, rpn_box/step_cnt, rcnn_cls/step_cnt, rcnn_box/step_cnt, sim_loss/step_cnt )
                    )
            re_cnt = True
        if use_tensorboard and cnt % log_interval == 0:
            exp.add_scalar_value('train_loss', train_loss / step_cnt, step=cnt)
            exp.add_scalar_value('learning_rate', lr, step=cnt)
            if _DEBUG:
                match_rate = net.match / net.set * 100. if cfg.TRIPLET.IS_TRUE else 0.
                triplet_loss = net.triplet_loss.data.cpu().numpy() if cfg.TRIPLET.IS_TRUE else 0.
                exp.add_scalar_value('true_positive', tp/fg*100., step=cnt)
                exp.add_scalar_value('true_negative', tf/bg*100., step=cnt)
                exp.add_scalar_value('true_distance', match_rate, step=cnt)
                losses = {'rpn_cls': float(rpn_cls/step_cnt),
                          'rpn_box': float(rpn_box/step_cnt),
                          'rcnn_cls': float(rcnn_cls/step_cnt),
                          'rcnn_box': float(rcnn_box/step_cnt),
                          'sim_loss': float(sim_loss/step_cnt)}
                exp.add_scalar_dict(losses, step=cnt)

        if re_cnt:
            train_loss = 0
            tp, tf, fg, bg, tp_box, fg_box = 0., 0., 0, 0, 0., 0
            rpn_cls, rpn_box, rcnn_cls, rcnn_box, sim_loss = 0., 0., 0., 0., 0.
            net.reset_match_count()
            step_cnt = 0
            t.tic()
            re_cnt = False

    # if epoch % save_interval == 0 and cnt > 0:
    save_dir = os.path.join(output_dir, model_name)
    make_dir(save_dir)
    save_name = os.path.join(save_dir, '{}_{}_{}_{}_b{}.h5'
                             .format(imdb_name, epoch, model_name, fg_thresh, batch_size))
    network.save_net(save_name, net)
    print('save model: {}'.format(save_name))
    if pf/tot > 70:
        print('Entering Test Phase ...')
        f = open('precision.txt', 'a')
        precision = test(save_name, net, test_imdb, test_roidb)
        match = id_match_test(save_name, net, test_imdb, test_roidb, cfg.TRIPLET.LOSS) if cfg.TRIPLET.IS_TRUE else 0.
        f.write(save_name + '  ----{:.2f}% / {:.2f}%\n'.format(precision, match))
        f.close()
        if previous_precision == 0.:
            previous_precision = precision
        else:
            if previous_precision > precision:
                print('Precision decreased {:.2f}% -> {:.2f}% ...' \
                      .format(previous_precision, precision))
                import warnings
                warnings.warn('test set Precision decreased. Keep Watching')
            else:
                previous_precision = precision
