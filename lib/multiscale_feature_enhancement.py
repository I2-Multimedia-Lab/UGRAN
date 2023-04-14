import torch
import torch.nn as nn
import torch.nn.functional as F

#from .layers import *
from .modules import *
class MFE(nn.Module):
    # Multilevel Feature Enhancement
    def __init__(self, in_channel, l_channel = None, h_channel = None, out_channel = 64, base_size=None, stage=None):
        super(MFE, self).__init__()
        self.relu = nn.ReLU(True)
        if base_size is not None and stage is not None:
            self.stage_size = (base_size[0] // (2 ** stage), base_size[1] // (2 ** stage))
        else:
            self.stage_size = None

        self.ci = Conv2d(in_channel,out_channel,3,relu=True)
        self.si = Conv2d(out_channel,out_channel,3)

        if h_channel != None:
            # channel transform
            self.ch = Conv2d(h_channel, out_channel,3,relu=True)

            # spatial transform
            self.sh = nn.Sequential(
                nn.Upsample(size=self.stage_size,mode='bilinear'),
                Conv2d(out_channel,out_channel,3)
                )
        
        if l_channel != None:
            # channel transform
            self.cl = Conv2d(l_channel, out_channel,3,relu=True)

            # spatial transform
            self.sl = nn.Sequential(
                nn.Upsample(size=self.stage_size,mode='bilinear'),
                Conv2d(out_channel,out_channel,3)
                )

        # diverse feature enhancement
        self.conv_asy = asyConv(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False)
        self.conv_atr = Conv2d(out_channel,out_channel,3,dilation=2)
        self.conv_ori = Conv2d(out_channel,out_channel,3,dilation=1)
        
        self.conv_cat = Conv2d(out_channel*3,out_channel,3)
        self.conv_res = Conv2d(in_channel, out_channel, 1)
        #self.initialize()

    def forward(self, x_i, x_l=None, x_h=None):
        x = self.ci(x_i)
        x = self.si(x)
        if x_h != None:
            x_h = self.ch(x_h)
            x_h = self.sh(x_h)
            x = x + x_h
        
        if x_l != None:
            x_l = self.cl(x_l)
            x_l = self.sl(x_l)
            x = x + x_l
        
        asy = self.conv_asy(x)
        atr = self.conv_atr(x)
        ori = self.conv_ori(x)
        x_cat = self.conv_cat(torch.cat((asy,atr,ori), dim = 1))
        x = self.relu(x_cat + self.conv_res(x_i))

        return x
    def initialize(self):
        weight_init(self)