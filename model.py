#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch 
import torch.nn as nn
from torchsummary import summary

from collections import OrderedDict
import math

class Con2D(nn.Module):
    def __init__(self, in_c, out_c, k_size, is_bn=True):
        super(Con2D, self).__init__()

        if is_bn:
            self.sequential = nn.Sequential(
                nn.Conv2d(in_c, out_c, k_size),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, k_size),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        else:
            self.sequential = nn.Sequential(
                nn.Conv2d(in_c, out_c, k_size),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, k_size),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.sequential(x)


def crop(features, size):
    h_old, w_old = features[0][0].size()
    h, w = size
    x = math.ceil((h_old - h) / 2)
    y = math.ceil((w_old - w) / 2)
    return features[:, :, x:(x + h), y:(y + w)]

class U_net(nn.Module):
    def __init__(self, in_channles, out_channels):
        super(U_net, self).__init__()

        self.con_block_1 = Con2D(in_channles, 64, 3)
        self.con_block_2 = Con2D(64, 128, 3)
        self.con_block_3 = Con2D(128, 256, 3)
        self.con_block_4 = Con2D(256, 512, 3)
        self.con_block_5 = Con2D(512, 1024, 3)

        self.exp_block_4 = Con2D(1024, 512, 3, is_bn=False)
        self.exp_block_3 = Con2D(512, 256, 3, is_bn=False)
        self.exp_block_2 = Con2D(256, 128, 3, is_bn=False)
        self.exp_block_1 = Con2D(128, 64, 3, is_bn=False)

        self.deconv_4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.deconv_3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.deconv_2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.deconv_1 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.final_layer = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        con_block_1_out = self.con_block_1(x)
        x = nn.MaxPool2d(2, stride=2)(con_block_1_out)
        con_block_2_out = self.con_block_2(x)
        x = nn.MaxPool2d(2, stride=2)(con_block_2_out)
        con_block_3_out = self.con_block_3(x)
        x = nn.MaxPool2d(2, stride=2)(con_block_3_out)
        con_block_4_out = self.con_block_4(x)
        x = nn.MaxPool2d(2, stride=2)(con_block_4_out)
        x = self.con_block_5(x)

        x = self.deconv_4(x)
        x = torch.cat([crop(con_block_4_out, (x.size()[2], x.size()[3])), x], dim=1)
        x = self.exp_block_4(x)

        x = self.deconv_3(x)
        x = torch.cat([crop(con_block_3_out,(x.size()[2], x.size()[3])), x], dim=1)
        x = self.exp_block_3(x)

        x = self.deconv_2(x)
        x = torch.cat([crop(con_block_2_out,(x.size()[2], x.size()[3])), x], dim=1)
        x = self.exp_block_2(x)

        x = self.deconv_1(x)
        x = torch.cat([crop(con_block_1_out,(x.size()[2], x.size()[3])), x], dim=1)
        x = self.exp_block_1(x)

        x = self.final_layer(x)

        return x


