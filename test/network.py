"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""
import torch
import torch.nn as nn


class MyResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels):
        super(MyResBlock, self).__init__()
        self.pad1 = nn.ReplicationPad2d(1)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True)
        self.in1 = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=True)
        self.relu = nn.ReLU(inplace=True)
        self.pad2 = nn.ReplicationPad2d(1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0, bias=True)
        self.in2 = nn.InstanceNorm2d(out_channels, affine=False, track_running_stats=True)

    def forward(self, x):
        residual = x
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.in1(out)
        out = self.relu(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.in2(out)
        out += residual
        out = self.relu(out)

        return out


class PixelAffinityNet(nn.Module):
    def __init__(self, nr_channel, conv1_size, use_canny=True):
        super(PixelAffinityNet, self).__init__()
        pad_size = (conv1_size - 1) / 2
        self.pad1 = nn.ReplicationPad2d((pad_size, pad_size, 0, 0))     # left, right, top, bottom
        self.conv1 = nn.Conv2d(3, nr_channel, kernel_size=(1, conv1_size), stride=1, padding=0, bias=True)
        self.in1 = nn.InstanceNorm2d(nr_channel, affine=False, track_running_stats=True)
        self.relu1 = nn.ReLU()
        self.res2 = MyResBlock(nr_channel, nr_channel)
        self.res3 = MyResBlock(nr_channel, nr_channel)
        self.res4 = MyResBlock(nr_channel, nr_channel)
        self.conv5 = nn.Conv2d(nr_channel, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.in5 = nn.InstanceNorm2d(1, affine=False, track_running_stats=True)
        self.sigm5 = nn.Sigmoid()

        self.use_canny = use_canny
        if self.use_canny:
            print('use_canny = True')
            self.conv6 = nn.Conv2d(2, 8, kernel_size=1, stride=1, bias=True)
            self.in6 = nn.InstanceNorm2d(8, affine=False, track_running_stats=True)
            self.relu6 = nn.ReLU()
            self.conv7 = nn.Conv2d(8, 1, kernel_size=1, stride=1, bias=True)
            self.in7 = nn.InstanceNorm2d(1, affine=False, track_running_stats=True)
            self.sigm7 = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.use_canny:
            image = self.pad1(x[:, 0:3, :, :])
        else:
            image = self.pad1(x)
        out = self.conv1(image)
        out = self.in1(out)
        out = self.relu1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.conv5(out)
        out = self.in5(out)
        out = self.sigm5(out)

        if self.use_canny:
            canny = x[:, 3:4, :, :]
            out = torch.cat((out, canny), 1)
            out = self.conv6(out)
            out = self.in6(out)
            out = self.relu6(out)
            out = self.conv7(out)
            out = self.in7(out)
            out = self.sigm7(out)

        return out
