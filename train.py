import os
import torch
import numpy as np
from torch.autograd import Variable
from datetime import datetime
from faster_rcnn import network
from faster_rcnn.network import init_data, data_to_variable
from faster_rcnn.faster_rcnn import FasterRCNN
from faster_rcnn.utils.timer import Timer

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


def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)



# hyper-parameters
# ------------
#imdb_name = 'voc_2007_trainval'
imdb_name = 'CaltechPedestrians'
cfg_file = 'experiments/cfgs/faster_rcnn_end2end.yml'
#pretrained_model = 'data/pretrained_model/VGG_imagenet.npy'
pretrained_model = 'data/pretrained_model/VGGnet_fast_rcnn_iter_70000.h5'
output_dir = 'models/saved_model3'

start_epoch = 1
end_epoch = 100
lr_decay_step = 5
lr_decay = 1./10

rand_seed = 1024
_DEBUG = True
use_tensorboard = False
remove_all_log = False   # remove all historical experiments in TensorBoard
exp_name = None # the previous experiment name in TensorBoard

# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)
batch_size = cfg.TRAIN.IMS_PER_BATCH
lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval = cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS
save_interval = cfg.TRAIN.SNAPSHOT_ITERS
# load data        # PASCAL VOC 2007 : Total 5011 images, 15662 objects
imdb, roidb, ratio_list, ratio_index = extract_roidb(imdb_name)
train_size = len(roidb)
sampler_batch = sampler(train_size, batch_size)
dataset = roibatchLoader(imdb, roidb, ratio_list, ratio_index, batch_size,
                                                        imdb.num_classes, training=True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         sampler=sampler_batch, num_workers=0)


# load net
net = FasterRCNN(classes=imdb.classes, debug=_DEBUG)
network.weights_normal_init(net, dev=0.01)
network.load_net_pedestrians(pretrained_model, net)
blob = init_data(is_cuda=True)

# set net to be prepared to train
net.cuda()

params = list(net.parameters())
# optimizer = torch.optim.Adam(params[-8:], lr=lr)
optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)


if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# tensorboad
use_tensorboard = use_tensorboard and CrayonClient is not None
if use_tensorboard:
    cc = CrayonClient(hostname='127.0.0.1')
    if remove_all_log:
        cc.remove_all_experiments()
    if exp_name is None:
        exp_name = datetime.now().strftime('vgg16_%m-%d_%H-%M')
        exp = cc.create_experiment(exp_name)
    else:
        exp = cc.open_experiment(exp_name)



iters_per_epoch = int(train_size / batch_size)
# training
train_loss = 0
step_cnt = 0
cnt = 0
re_cnt = False
t = Timer()
t.tic()

for epoch in range(start_epoch, end_epoch+1):

    tp, tf, fg, bg = 0., 0., 0, 0
    net.train()
    if epoch % lr_decay_step == 0:
        lr *= lr_decay
        optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)
    data_iter = iter(dataloader)
    for step in range(iters_per_epoch):

        # get one batch
        data = next(data_iter)
        (im_data, im_info, gt_boxes, num_boxes) = data_to_variable(blob, data)

        # forward
        net.zero_grad()
        net(im_data, im_info, gt_boxes, num_boxes)
        loss = net.loss + net.rpn.loss
        # if np.isnan(float(loss.data[0])):
        #     import cv2
        #     print(im_data.data)
        #     print(im_info.data)
        #     print(gt_boxes.data)
        #     print(num_boxes.data)
        #     img_id = data[4]
        #     img = imdb.image_path_at(img_id)
        #     cv2.imshow(str(id), img)
        #     cv2.waitKey()
        #     cv2.destroyAllWindows()
        #     raise RuntimeError
        if _DEBUG:
            tp += float(net.tp)
            tf += float(net.tf)
            fg += net.fg_cnt
            bg += net.bg_cnt

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
                    log_print('\tTP: %.2f%%, TF: %.2f%%, fg/bg=(%d/%d)' % (tp/fg*100., tf/bg*100., fg/step_cnt, bg/step_cnt))
                    log_print('\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box: %.4f' % (
                        net.rpn.cross_entropy.data.cpu().numpy(), net.rpn.loss_box.data.cpu().numpy(),
                        net.cross_entropy.data.cpu().numpy(), net.loss_box.data.cpu().numpy())
                    )
            re_cnt = True

        if use_tensorboard and cnt % log_interval == 0:
            exp.add_scalar_value('train_loss', train_loss / step_cnt, step=cnt)
            exp.add_scalar_value('learning_rate', lr, step=cnt)
            if _DEBUG:
                exp.add_scalar_value('true_positive', tp/fg*100., step=cnt)
                exp.add_scalar_value('true_negative', tf/bg*100., step=cnt)
                losses = {'rpn_cls': float(net.rpn.cross_entropy.data.cpu().numpy()[0]),
                          'rpn_box': float(net.rpn.loss_box.data.cpu().numpy()[0]),
                          'rcnn_cls': float(net.cross_entropy.data.cpu().numpy()[0]),
                          'rcnn_box': float(net.loss_box.data.cpu().numpy()[0])}
                exp.add_scalar_dict(losses, step=cnt)

        if cnt % save_interval == 0 and cnt > 0:
            save_name = os.path.join(output_dir, 'faster_rcnn_pedestrians{}_b{}.h5'.format(cnt , batch_size))
            network.save_net(save_name, net)
            print('save model: {}'.format(save_name))

        if re_cnt:
            train_loss = 0
            tp, tf, fg, bg = 0., 0., 0, 0
            step_cnt = 0
            t.tic()
            re_cnt = False


