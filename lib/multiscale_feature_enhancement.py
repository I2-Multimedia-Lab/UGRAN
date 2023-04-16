import torch
import torch.nn as nn
import torch.nn.functional as F

#from .layers import *
from .modules import *
class MFE0(nn.Module):
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
        
        self.forward = self._forward
    def initialize(self):
        weight_init(self)

    def _forward(self, x_i, x_l=None, x_h=None):
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
    
    def _ablation(self, x_i, x_l=None, x_h=None):

        x = self.conv_res(x_i)
        return x

class MFE1(nn.Module):

    def __init__(self, in_channel, l_channel = None, h_channel = None, out_channel = 64, base_size=None, stage=None, stride=1, scale = 0.1):
        super(MFE, self).__init__()
        self.scale = scale
        self.out_channel = out_channel
        inter_channel = in_channel //4


        self.branch0 = nn.Sequential(
                Conv2d(in_channel, inter_channel, kernel_size=1, stride=1,relu=True),
                Conv2d(inter_channel, inter_channel, kernel_size=3, stride=1,relu=False)
                )
        self.branch1 = nn.Sequential(
                Conv2d(in_channel, inter_channel, kernel_size=1, stride=1,relu=True),
                Conv2d(inter_channel, inter_channel, kernel_size=(3,1), stride=1,relu=True),
                Conv2d(inter_channel, inter_channel, kernel_size=3, stride=1, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                Conv2d(in_channel, inter_channel, kernel_size=1, stride=1,relu=True),
                Conv2d(inter_channel, inter_channel, kernel_size=(1,3), stride=stride, relu=True),
                Conv2d(inter_channel, inter_channel, kernel_size=3, stride=1, dilation=3, relu=False)
                )
        self.branch3 = nn.Sequential(
                Conv2d(in_channel, inter_channel//2, kernel_size=1, stride=1,relu=True),
                Conv2d(inter_channel//2, (inter_channel//4)*3, kernel_size=(1,3), stride=1,relu=True),
                Conv2d((inter_channel//4)*3, inter_channel, kernel_size=(3,1), stride=stride,relu=True),
                Conv2d(inter_channel, inter_channel, kernel_size=3, stride=1, dilation=5, relu=False)
                )

        self.ConvLinear = Conv2d(4*inter_channel, out_channel, kernel_size=1, stride=1, relu=False)
        self.shortcut = Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)
    def initialize(self):
        weight_init(self)
    def forward(self,x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0,x1,x2,x3),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)

        return out
    
class MFE2(nn.Module):
    # Multilevel Feature Enhancement
    def __init__(self, in_channel, l_channel = None, h_channel = None, out_channel = 64, base_size=None, stage=None):
        super(MFE, self).__init__()
        self.relu = nn.ReLU(True)
        if base_size is not None and stage is not None:
            self.stage_size = (base_size[0] // (2 ** stage), base_size[1] // (2 ** stage))
        else:
            self.stage_size = None

        #self.ci = Conv2d(in_channel,out_channel,3,relu=True)
        #self.si = Conv2d(out_channel,out_channel,3)

        if h_channel != None:
            # channel transform
            self.ch = Conv2d(h_channel, in_channel,3,relu=True)

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
        inter_channel = in_channel // 4


        self.branch0 = nn.Sequential(
                Conv2d(in_channel, inter_channel, kernel_size=1, stride=1,relu=True),
                Conv2d(inter_channel, inter_channel, kernel_size=3, stride=1,relu=False)
                )
        self.branch1 = nn.Sequential(
                Conv2d(in_channel, inter_channel, kernel_size=1, stride=1,relu=True),
                asyConv(in_channels=inter_channel, out_channels=inter_channel, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False),
                nn.ReLU(),
                Conv2d(inter_channel, inter_channel, kernel_size=3, stride=1, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                Conv2d(in_channel, inter_channel, kernel_size=1, stride=1,relu=True),
                asyConv(in_channels=inter_channel, out_channels=inter_channel, kernel_size=5, stride=1, padding=2, dilation=1, groups=1, padding_mode='zeros', deploy=False),
                nn.ReLU(),
                Conv2d(inter_channel, inter_channel, kernel_size=3, stride=1,  dilation=5, relu=False)
                )


        self.ConvLinear = Conv2d(3*inter_channel, out_channel, kernel_size=1, stride=1, relu=False)
        self.shortcut = Conv2d(in_channel, out_channel, kernel_size=1, stride=1, relu=False)
        self.relu = nn.ReLU(inplace=False)

        
        self.forward = self._forward
    def initialize(self):
        weight_init(self)

    def _forward(self, x_i, x_l=None, x_h=None):
        x = x_i
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = self.relu(out + short)
        
        return out
    
    def _ablation(self, x_i, x_l=None, x_h=None):

        x = self.conv_res(x_i)
        return x
    
class MFE(nn.Module):
    """ Enhance the feature diversity.
    """
    def __init__(self, in_channel, l_channel = None, h_channel = None, out_channel = 64, base_size=None, stage=None):
        super(MFE, self).__init__()
        self.asyConv = asyConv(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False)
        self.oriConv = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.atrConv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, dilation=2, padding=2, stride=1), nn.BatchNorm2d(out_channel), nn.ReLU()
        )           
        self.conv2d = nn.Conv2d(out_channel*3, out_channel, kernel_size=3, stride=1, padding=1)
        self.bn2d = nn.BatchNorm2d(out_channel)
        self.initialize()

    def forward(self, f):
        p1 = self.oriConv(f)
        p2 = self.asyConv(f)
        p3 = self.atrConv(f)
        p  = torch.cat((p1, p2, p3), 1)
        p  = F.relu(self.bn2d(self.conv2d(p)), inplace=True)

        return p

    def initialize(self):
        #pass
        weight_init(self)