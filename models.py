#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

from torch import nn
import torch.nn.functional as F
# 所有神经网络的基类torch.nn
from conv2d import ConvBlock
# conv+bn+relu


class AlexNetNormal(nn.Module):
    def __init__(self, args, in_channels = 3, num_classes = 10, norm_type='bn'):
        super(AlexNetNormal, self).__init__()

        self.features = nn.Sequential(
            ConvBlock(in_channels, 64, 5, 1, 2, bn=norm_type),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16
            ConvBlock(64, 192, 5, 1, 2, bn=norm_type),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8x8
            ConvBlock(192, 384, bn=norm_type),
            ConvBlock(384, 256, bn=norm_type),
            ConvBlock(256, 256, bn=norm_type),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4x4
        )
        # if args.dataset == 'cifar100':
        #     num_classes = 100
        self.classifier = nn.Linear(4 * 4 * 256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class VGG(nn.Module):
    def __init__(self, args, in_channels = 3, num_classes = 10, norm_type='bn'):
        super(VGG, self).__init__()

        self.features = nn.Sequential(
            ConvBlock(in_channels, 64, 3, 1, 1, bn=norm_type),
            ConvBlock(64, 64, 3, 1, 1, bn=norm_type),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),  # 16x16
            ConvBlock(64, 128, 3, 1, 1, bn=norm_type),
            ConvBlock(128, 128, 3, 1, 1, bn=norm_type),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),  # 8x8
            ConvBlock(128, 256, 3, 1, 1, bn=norm_type),
            ConvBlock(256, 256, 3, 1, 1, bn=norm_type),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            ConvBlock(256, 512, 3, 1, 1, bn=norm_type),
            ConvBlock(512, 512, 3, 1, 1, bn=norm_type),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),
            ConvBlock(512, 512, 3, 1, 1, bn=norm_type),
            ConvBlock(512, 512, 3, 1, 1, bn=norm_type),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1),  # 4x4
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        if args.dataset == 'cifar100':
            num_classes = 100

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


import math
from blocks import BaseBlock

class MobileNetV2(nn.Module):
    def __init__(self, args, output_size=10, alpha = 1):
        super(MobileNetV2, self).__init__()
        self.output_size = output_size

        # first conv layer 
        self.conv0 = nn.Conv2d(3, int(32*alpha), kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn0 = nn.BatchNorm2d(int(32*alpha))

        # build bottlenecks
        BaseBlock.alpha = alpha
        self.bottlenecks = nn.Sequential(
            BaseBlock(32, 16, t = 1, downsample = False),
            BaseBlock(16, 24, downsample = False),
            BaseBlock(24, 24),
            BaseBlock(24, 32, downsample = False),
            BaseBlock(32, 32),
            BaseBlock(32, 32),
            BaseBlock(32, 64, downsample = True),
            BaseBlock(64, 64),
            BaseBlock(64, 64),
            BaseBlock(64, 64),
            BaseBlock(64, 96, downsample = False),
            BaseBlock(96, 96),
            BaseBlock(96, 96),
            BaseBlock(96, 160, downsample = True),
            BaseBlock(160, 160),
            BaseBlock(160, 160),
            BaseBlock(160, 320, downsample = False))

        if args.dataset == 'cifar100':
            output_size = 100

        # last conv layers and fc layer
        self.conv1 = nn.Conv2d(int(320*alpha), 1280, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(1280)
        self.fc = nn.Linear(1280, output_size)

    def forward(self, inputs):

        # first conv layer
        x = F.relu6(self.bn0(self.conv0(inputs)), inplace = True)
        # assert x.shape[1:] == torch.Size([32, 32, 32])

        # bottlenecks
        x = self.bottlenecks(x)
        # assert x.shape[1:] == torch.Size([320, 8, 8])

        # last conv layer
        x = F.relu6(self.bn1(self.conv1(x)), inplace = True)
        # assert x.shape[1:] == torch.Size([1280,8,8])

        # global pooling and fc (in place of conv 1x1 in paper)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_type='bn'):
        super(BasicBlock, self).__init__()

        self.convbnrelu_1 = ConvBlock(in_planes, planes, 3, stride, 1, bn=norm_type, relu=True)
        self.convbn_2 = ConvBlock(planes, planes, 3, 1, 1, bn=norm_type, relu=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = ConvBlock(in_planes, self.expansion * planes,
                                      1, stride, 0, bn=norm_type, relu=True)

    def forward(self, x):
        out = self.convbnrelu_1(x)
        out = self.convbn_2(out)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm_type='bn'):
        super(Bottleneck, self).__init__()

        self.convbnrelu_1 = ConvBlock(in_planes, planes, 1, 1, 0, bn=norm_type, relu=True)
        self.convbnrelu_2 = ConvBlock(planes, planes, 3, stride, 1, bn=norm_type, relu=True)
        self.convbn_3 = ConvBlock(planes, self.expansion * planes, 1, 1, 0, bn=norm_type, relu=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = ConvBlock(in_planes, self.expansion * planes, 1, stride, 0, bn=norm_type, relu=False)

    def forward(self, x):
        out = self.convbnrelu_1(x)
        out = self.convbnrelu_2(out)
        out = self.convbn_3(out) + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100, norm_type='bn', pretrained=False, imagenet=False):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.num_blocks = num_blocks
        self.norm_type = norm_type

        self.convbnrelu_1 = ConvBlock(3, 64, 3, 1, 1, bn=norm_type, relu=True)  # 32
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)  # 32/ 56
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # 16/ 28
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # 8/ 14
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # 4/ 7
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm_type))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.convbnrelu_1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def ResNet9(**model_kwargs):
    return ResNet(BasicBlock, [1, 1, 1, 1], **model_kwargs)


def ResNet18(**model_kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **model_kwargs)


def ResNet34(**model_kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **model_kwargs)


def ResNet50(**model_kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **model_kwargs)


def ResNet101(**model_kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **model_kwargs)


def ResNet152(**model_kwargs):
    return ResNet(Bottleneck, [3, 8, 36, 3], **model_kwargs)

class WideBasic(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )

        self.shortcut = nn.Sequential()

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride)
            )

    def forward(self, x):

        residual = self.residual(x)
        shortcut = self.shortcut(x)

        return residual + shortcut

class WideResNet(nn.Module):
    def __init__(self, num_classes, block, depth=50, widen_factor=1):
        super().__init__()

        self.depth = depth
        k = widen_factor
        l = int((depth - 4) / 6)
        self.in_channels = 16
        self.init_conv = nn.Conv2d(3, self.in_channels, 3, 1, padding=1)
        self.conv2 = self._make_layer(block, 16 * k, l, 1)
        self.conv3 = self._make_layer(block, 32 * k, l, 2)
        self.conv4 = self._make_layer(block, 64 * k, l, 2)
        self.bn = nn.BatchNorm2d(64 * k)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(64 * k, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        return x

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

def wideresnet(depth=40, widen_factor=10):
    net = WideResNet(100, WideBasic, depth=depth, widen_factor=widen_factor)
    return net