import torch
import torch.nn as nn
import torch.nn.functional as F

#from .layers import *
from .modules import *
class MFE0(nn.Module):
    # ori MFE
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
        
        self.forward = self._ablation
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
    # RFB
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
    
class MFE(nn.Module):
    # MFE with RFB+atrconv
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
            self.ch = Conv2d(h_channel, in_channel,1,relu=True)

            # spatial transform
            self.sh = nn.Sequential(
                nn.Upsample(size=self.stage_size,mode='bilinear'),
                #Conv2d(out_channel,out_channel,3)
                )
        
        if l_channel != None:
            # channel transform
            self.cl = Conv2d(l_channel, in_channel,1,relu=True)

            # spatial transform
            self.sl = nn.Sequential(
                nn.Upsample(size=self.stage_size,mode='bilinear'),
                #Conv2d(out_channel,out_channel,3)
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
                Conv2d(inter_channel, inter_channel, kernel_size=3, stride=1, dilation=5, relu=False)
                )

        self.ConvLinear = Conv2d(3*inter_channel, out_channel, kernel_size=1, stride=1, relu=False)
        #self.channel_att = SE(3*inter_channel)
        self.shortcut = Conv2d(in_channel, out_channel, kernel_size=1, stride=1, relu=False)
        self.relu = nn.ReLU(inplace=False)
        
        self.forward = self._ablation

    def initialize(self):
        weight_init(self)

    def _forward(self, x_i, x_l=None, x_h=None):
        x = x_i
        if x_h != None:
            x_h = self.ch(x_h)
            x_h = self.sh(x_h)
            x = x + x_h
        
        if x_l != None:
            x_l = self.cl(x_l)
            x_l = self.sl(x_l)
            x = x + x_l
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0,x1,x2),1)
        #out = self.channel_att(out)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = self.relu(out + short)
        
        return out
    
    def _ablation(self, x_i, x_l=None, x_h=None):

        x = self.shortcut(x_i)
        return x
    
class MFE3(nn.Module):
    # DFA
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

            
class MFE4(nn.Module):
    # PSP
    def __init__(self, in_channel, l_channel = None, h_channel = None, out_channel = 64, base_size=None, stage=None):
        super(MFE, self).__init__()
        reduction_dim = in_channel//4
        bins = [1,2,3,6]
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_channel, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)
        self.ConvLinear = Conv2d(in_channel+4*reduction_dim, out_channel, kernel_size=1, stride=1, relu=True)
    def initialize(self):
        #pass
        weight_init(self)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        out = torch.cat(out, 1)
        return self.ConvLinear(out)
    
class MFE5(nn.Module):
    # inception
    def __init__(self, in_channel, l_channel = None, h_channel = None, out_channel = 64, base_size=None, stage=None):
        layers=[64,48,64,64,96,96,32]
        super(MFE,self).__init__()
        self.branch1 = nn.Sequential(
        nn.Conv2d(in_channel,layers[0],kernel_size=1,stride=1,padding=0 ,bias=False),
        nn.BatchNorm2d(layers[0]),
        nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
        nn.Conv2d(in_channel,layers[1],1,stride=1,padding=0,bias=False),
        nn.BatchNorm2d(layers[1]),
        nn.ReLU(inplace=True),
        nn.Conv2d(layers[1],layers[2],5,stride=1,padding=2,bias=False),
        nn.BatchNorm2d(layers[2]),
        nn.ReLU(True),
        )
        self.branch3 = nn.Sequential(
        nn.Conv2d(in_channel,layers[3],kernel_size=1,stride=1,padding=0,bias=False),
        nn.BatchNorm2d(layers[3]),
        nn.ReLU(inplace=True),
        nn.Conv2d(layers[3],layers[4],kernel_size=3,stride=1,padding=1,bias=False),
        nn.BatchNorm2d(layers[4]),
        nn.ReLU(inplace=True),
        nn.Conv2d(layers[4],layers[5],kernel_size=3,stride=1,padding=1,bias=False),
        nn.BatchNorm2d(layers[5]),
        nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
        nn.AvgPool2d(kernel_size=3,stride=1,padding=1),
        nn.Conv2d(in_channel,layers[6],1,stride=1,padding=0,bias=False),
        nn.BatchNorm2d(layers[6]),
        nn.ReLU(inplace=True),
        )
        self.ConvLinear = Conv2d(layers[0]+layers[2]+layers[5]+layers[6], out_channel, kernel_size=1, stride=1, relu=True)
    def initialize(self):
        #pass
        weight_init(self)

    def forward(self,x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        out = torch.cat([b1,b2,b3,b4],dim=1)
        return self.ConvLinear(out)
    
class MFE6(nn.Module):
    # ASPP
    def __init__(self, in_channel, l_channel = None, h_channel = None, out_channel = 64, base_size=None, stage=None):
        super(MFE,self).__init__()
        depth = in_channel // 4
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
 
        self.conv_1x1_output = nn.Conv2d(depth * 5, out_channel, 1, 1)
    def initialize(self):
        #pass
        weight_init(self)

    def forward(self, x):
        size = x.shape[2:]
 
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')
 
        atrous_block1 = self.atrous_block1(x)
 
        atrous_block6 = self.atrous_block6(x)
 
        atrous_block12 = self.atrous_block12(x)
 
        atrous_block18 = self.atrous_block18(x)
 
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net

class MFE7(nn.Module):
    # SE test
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
            self.ch = Conv2d(h_channel, in_channel,1,relu=True)

            # spatial transform
            self.sh = nn.Sequential(
                nn.Upsample(size=self.stage_size,mode='bilinear'),
                #Conv2d(out_channel,out_channel,3)
                )
        
        if l_channel != None:
            # channel transform
            self.cl = Conv2d(l_channel, out_channel,1,relu=True)

            # spatial transform
            self.sl = nn.Sequential(
                nn.Upsample(size=self.stage_size,mode='bilinear'),
                #Conv2d(out_channel,out_channel,3)
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
                )
        self.branch2 = nn.Sequential(
                Conv2d(in_channel, inter_channel, kernel_size=1, stride=1,relu=True),
                asyConv(in_channels=inter_channel, out_channels=inter_channel, kernel_size=5, stride=1, padding=2, dilation=1, groups=1, padding_mode='zeros', deploy=False),
                nn.ReLU(),
                )
        self.branch3 = nn.Sequential(
                Conv2d(in_channel, inter_channel, kernel_size=1, stride=1,relu=True),
                Conv2d(inter_channel, inter_channel, kernel_size=3, stride=1, dilation=2, relu=False)
                )
        self.branch4 = nn.Sequential(
                Conv2d(in_channel, inter_channel, kernel_size=1, stride=1,relu=True),
                Conv2d(inter_channel, inter_channel, kernel_size=3, stride=1, dilation=4, relu=False)
                )

        self.ConvLinear = Conv2d(5*inter_channel, out_channel, kernel_size=1, stride=1, relu=False)
        self.channel_att = GCT(5*inter_channel)
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
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        out = torch.cat((x0,x1,x2,x3,x4),1)
        out,w = self.channel_att(out)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = self.relu(out + short)
        
        return out,w
    def _ablation(self, x_i, x_l=None, x_h=None):

        x = self.shortcut(x_i)
        return x
    
class MFE_l(nn.Module):
    # MFE_l
    # Multilevel Feature Enhancement
    def __init__(self, in_channel, l_channel = None, h_channel = None, out_channel = 64, base_size=None, stage=None):
        super(MFE, self).__init__()
        self.relu = nn.ReLU(True)
        if base_size is not None and stage is not None:
            self.stage_size = (base_size[0] // (2 ** stage), base_size[1] // (2 ** stage))
        else:
            self.stage_size = None

        if h_channel != None:
            # channel transform
            self.ch = Conv2d(h_channel, in_channel,1,relu=True)

            # spatial transform
            self.sh = nn.Sequential(
                nn.Upsample(size=self.stage_size,mode='bilinear'),
                #Conv2d(out_channel,out_channel,3)
                )
        
        if l_channel != None:
            # channel transform
            self.cl = Conv2d(l_channel, out_channel,1,relu=True)

            # spatial transform
            self.sl = nn.Sequential(
                nn.Upsample(size=self.stage_size,mode='bilinear'),
                #Conv2d(out_channel,out_channel,3)
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
                #nn.ReLU(),
                )
        self.branch2 = nn.Sequential(
                Conv2d(in_channel, inter_channel, kernel_size=1, stride=1,relu=True),
                Conv2d(inter_channel, inter_channel, kernel_size=3, stride=1, dilation=2, relu=False)
                )

        self.ConvLinear = Conv2d(3*inter_channel, out_channel, kernel_size=1, stride=1, relu=False)
        #self.channel_att = GCT(3*inter_channel)
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
        #out = self.channel_att(out)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = self.relu(out + short)
        
        return out
    def _ablation(self, x_i, x_l=None, x_h=None):

        x = self.shortcut(x_i)
        return x

class MFE_h(nn.Module):
    # mfe_h
    # Multilevel Feature Enhancement
    def __init__(self, in_channel, l_channel = None, h_channel = None, out_channel = 64, base_size=None, stage=None):
        super(MFE, self).__init__()
        self.relu = nn.ReLU(True)
        if base_size is not None and stage is not None:
            self.stage_size = (base_size[0] // (2 ** stage), base_size[1] // (2 ** stage))
        else:
            self.stage_size = None

        if h_channel != None:
            # channel transform
            self.ch = Conv2d(h_channel, in_channel,1,relu=True)

            # spatial transform
            self.sh = nn.Sequential(
                nn.Upsample(size=self.stage_size,mode='bilinear'),
                #Conv2d(out_channel,out_channel,3)
                )
        
        if l_channel != None:
            # channel transform
            self.cl = Conv2d(l_channel, out_channel,1,relu=True)

            # spatial transform
            self.sl = nn.Sequential(
                nn.Upsample(size=self.stage_size,mode='bilinear'),
                #Conv2d(out_channel,out_channel,3)
                )

        # diverse feature enhancement
        inter_channel = in_channel // 4

        self.interaction = nn.Sequential(
                SE(in_channel),
                Conv2d(in_channel, inter_channel, kernel_size=1, stride=1,relu=True),
                Conv2d(inter_channel, out_channel, kernel_size=3, stride=1,relu=True),
                )

   
        #self.channel_att = GCT(inter_channel)
        self.shortcut = nn.Sequential(
                SE(in_channel), 
                Conv2d(in_channel, out_channel, kernel_size=1, stride=1, relu=False)
        )
        self.relu = nn.ReLU(inplace=False)
        
        self.forward = self._forward

    def initialize(self):
        weight_init(self)

    def _forward(self, x_i, x_l=None, x_h=None):
        x = x_i
        
        x = self.interaction(x)

        return x
        
        #return out
    def _ablation(self, x_i, x_l=None, x_h=None):

        x = self.shortcut(x_i)
        return x