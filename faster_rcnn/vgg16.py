from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
import pdb


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.model_path = 'data/pretrained_model/base/vgg16_caffe.pth'
        vgg = models.vgg16()
        self.vgg = vgg
        self.vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

        # not using the last maxpool layer
        self.features = nn.Sequential(*list(vgg.features._modules.values())[:-1])

        # Fix the layers before conv3:
        for layer in range(10):
            for p in self.features[layer].parameters(): p.requires_grad = False

        self.fc_layer = self.vgg.classifier
        self.load_pretrained_vgg16 = self._load_vgg16

    def _load_vgg16(self):
        print("Loading pretrained weights from %s" % (self.model_path))
        state_dict = torch.load(self.model_path)
        self.vgg.load_state_dict({k: v for k, v in state_dict.items() if k in self.vgg.state_dict()})
        del self.vgg
    def forward(self, im_data):
        x = self.features(im_data)
        return x