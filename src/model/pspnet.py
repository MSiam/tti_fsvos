import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .resnet import resnet50, resnet101
from .vgg import vgg16_bn
from .video_swin import build_swin_b_backbone

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


def get_vgg16_layer(model):
    layer0_idx = range(0, 7)
    layer1_idx = range(7, 14)
    layer2_idx = range(14, 24)
    layer3_idx = range(24, 34)
    layer4_idx = range(34, 43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]
    layer0 = nn.Sequential(*layers_0)
    layer1 = nn.Sequential(*layers_1)
    layer2 = nn.Sequential(*layers_2)
    layer3 = nn.Sequential(*layers_3)
    layer4 = nn.Sequential(*layers_4)
    return layer0, layer1, layer2, layer3, layer4


class PSPNet(nn.Module):
    def __init__(self, args, zoom_factor, use_ppm):
        super(PSPNet, self).__init__()
        # assert args.layers in [50, 101, 152]
        assert 2048 % len(args.bins) == 0
        assert args.num_classes_tr > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = use_ppm
        self.m_scale = args.m_scale
        self.bottleneck_dim = args.bottleneck_dim
        self.multires_classifier = False if not hasattr(args, 'multires_classifier') else args.multires_classifier
        self.classifier_chs = [self.bottleneck_dim, 256]

        self.arch = args.arch
        if args.arch == 'resnet':
            if args.layers == 50:
                resnet = resnet50(pretrained=args.pretrained)
            else:
                resnet = resnet101(pretrained=args.pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3,
                                        resnet.bn3, resnet.relu, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
            self.feature_res = (53, 53)
        elif args.arch == 'vgg':
            vgg = vgg16_bn(pretrained=args.pretrained)
            self.layer0, self.layer1, self.layer2, self.layer3, self.layer4 = get_vgg16_layer(vgg)
        elif args.arch == 'videoswin':
            self.videoswin_backbone = build_swin_b_backbone(
                "/local/riemann/home/rezaul/projects/medvt2-main/pretrained/swin_base_patch244_window877_kinetics400_22k.pth"
            )
            self.feature_res = (8, 14)

        if self.arch != "videoswin":
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        if self.m_scale:
            fea_dim = 1024 + 512
        else:
            if args.arch == 'resnet':
                fea_dim = 2048
            elif args.arch == 'vgg':
                fea_dim = 512
            elif args.arch == 'videoswin':
                fea_dim = 1024

        if use_ppm:
            self.ppm = PPM(fea_dim, int(fea_dim/len(args.bins)), args.bins)
            fea_dim *= 2

        self.bottleneck = nn.Sequential(
            nn.Conv2d(fea_dim, self.bottleneck_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=args.dropout))

        if self.multires_classifier:
            self.multires_mixer = nn.Sequential(
                nn.Conv2d(self.classifier_chs[0] + self.classifier_chs[1], self.bottleneck_dim, kernel_size=1),
                nn.BatchNorm2d(self.bottleneck_dim),
                nn.ReLU(inplace=True))
        self.classifier = nn.ModuleList([nn.Conv2d(self.bottleneck_dim, args.num_classes_tr, kernel_size=1)])

    def get_backbone_modules(self):
        if self.arch == "videoswin":
            return [self.videoswin_backbone]
        else:
            return [self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]

    def get_new_modules(self):
        if self.use_ppm:
            return [self.ppm, self.bottleneck, self.classifier]
        else:
            return [self.bottleneck, self.classifier]

    def set_feature_res(self, size):
        self.feature_res = (int(np.ceil(size[0]/8.0)), int(np.ceil(size[1]/8.0)))

    def freeze_bn(self):
        for m in self.modules():
            if not isinstance(m, nn.BatchNorm2d):
                m.eval()

    def forward(self, x):
        x_size = x.size()
        assert (x_size[-2]-1) % 8 == 0 and (x_size[-1]-1) % 8 == 0
        H = int((x_size[-2] - 1) / 8 * self.zoom_factor + 1)
        W = int((x_size[-1] - 1) / 8 * self.zoom_factor + 1)

        x = self.extract_features(x)
        x = self.classify(x, (H, W))
        return x

    def extract_features(self, x):
        if self.arch == 'videoswin':
            x = self.videoswin_backbone(x.permute(0,2,1,3,4))[-1]
            x = x.permute(0,2,1,3,4)
            x = x.view(-1, *x.shape[-3:]).contiguous()
        else:
            x = self.layer0(x)
            x_1 = self.layer1(x)
            x_2 = self.layer2(x_1)
            x_3 = self.layer3(x_2)
            if self.m_scale:
                x = torch.cat([x_2, x_3], dim=1)
            else:
                x = self.layer4(x_3)

        if self.use_ppm:
            x = self.ppm(x)

        x = self.bottleneck(x)

        if self.multires_classifier:
            x = F.interpolate(x, x_1.shape[-2:], mode='bilinear', align_corners=True)
            x = self.multires_mixer(torch.cat([x, x_1], dim=1))
            return [x]
        else:
            return [x]

    def classify(self, features, shape):
#        probas = []
#        for i in range(len(self.classifier)):
#            x = self.classifier[i](features[i])
#            probas.append(x)
#        hr_res = probas[-1].shape[-2:]
#        probas = [F.interpolate(proba, hr_res) for proba in probas]
#        x = torch.stack(probas, dim=0).mean(dim=0)

        x = self.classifier[-1](features[-1])
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=shape, mode='bilinear', align_corners=True)
        return x
