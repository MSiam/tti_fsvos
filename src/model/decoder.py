#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Haoxin Chen
# @File    : decoder.py
# Code borrowed from DANet: https://github.com/scutpaul/DANet/blob/main/libs/models/DAN/decoder.py

import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r

class Refine(nn.Module):
    def __init__(self, inplanes, planes):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, size=s.shape[2:], mode='bilinear', align_corners=True)
        m = self.ResMM(m)
        return m

class Decoder(nn.Module):
    def __init__(self, inplane, mdim):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(inplane, mdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF2 = Refine(256, mdim)
        #self.pred2 = nn.Conv2d(mdim, num_classes, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, r4, r2):
        m4 = self.ResMM(self.convFM(r4))
        m2 = self.RF2(r2, m4)
        #p2 = self.pred2(F.relu(m2))
        #p = F.interpolate(p2, size=original_shape, mode='bilinear', align_corners=True)
        return F.relu(m2)
