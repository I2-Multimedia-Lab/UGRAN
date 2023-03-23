import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import *
class MFE(nn.Module):
    # Multilevel Feature Enhancement
    def __init__(self, in_channel, f_channel = None, out_channel = 64, base_size=None, stage=None):
        super(MFE, self).__init__()
        self.relu = nn.ReLU(True)
        if base_size is not None and stage is not None:
            self.stage_size = (base_size[0] // (2 ** stage), base_size[1] // (2 ** stage))
        else:
            self.stage_size = None

        if f_channel != None:
            # channel transform
            self.ci = Conv2d(in_channel,in_channel,3,relu=True)
            self.cf = Conv2d(f_channel, in_channel,3,relu=True)

            # spatial transform
            self.si = Conv2d(in_channel,in_channel,3)
            self.sf = nn.Sequential(
                nn.Upsample(size=self.stage_size,mode='bilinear'),
                Conv2d(in_channel,in_channel,3)
                )

        else:
            self.ci = Conv2d(in_channel,in_channel,3,relu=True)
            self.si = Conv2d(in_channel,in_channel,3)

        # diverse feature enhancement
        self.conv_asy = asyConv(in_channel,out_channel,3)
        self.conv_atr = Conv2d(in_channel,out_channel,3,dilation=2)
        self.conv_ori = Conv2d(in_channel,out_channel,3,dilation=1)
        
        self.conv_cat = Conv2d(out_channel*3,out_channel,3)
        self.conv_res = Conv2d(in_channel, out_channel, 1)

    def forward(self, x_i, x_f=None):
        x = self.ci(x_i)
        x = self.si(x)
        if x_f != None:
            x_f = self.cf(x_f)
            x_f = self.sf(x_f)
            x = x + x_f
        
        asy = self.conv_asy(x)
        atr = self.conv_atr(x)
        ori = self.conv_ori(x)
        x_cat = self.conv_cat(torch.cat((asy,atr,ori), dim = 1))
        x = self.relu(x_cat + self.conv_res(x_i))

        return x
