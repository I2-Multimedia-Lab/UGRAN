import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.modules import *
from .layers import *
class SSCA(nn.Module):
    # Scale Spatial Consistent Attention
    def __init__(self, in_channel, dim, depth=64, base_size=[384,384], stage=1):
        super(SSCA, self).__init__()
        
        if base_size is not None and stage is not None:
            self.stage_size = (base_size[0] // 16, base_size[1] // 16)
        else:
            self.stage_size = None
        self.ratio = 2**(4-stage)
        self.depth = depth
        self.dim = dim
        self.spatial_reduce = Conv2d(dim,dim,kernel_size=self.ratio,stride=self.ratio,padding='valid')
        #self.norm = nn.BatchNorm2d(depth)
        self.channel_trans = Conv2d(in_channel,dim,3,relu=True)
        self.q = nn.Sequential(
            Conv2d(dim,dim,3,relu=True),
        )
        self.k = nn.Sequential(
            Conv2d(dim,dim,1,relu=True),
        )
        self.v = nn.Sequential(
            Conv2d(dim,dim,1,relu=True),
        )

        self.conv_out1 = Conv2d(dim,depth,1,relu=True)
        self.conv_out2 = Conv2d(dim+depth, depth, 3, relu=True)
        self.conv_out3 = Conv2d(depth, depth, 3, relu=True)
    def forward(self, x):

        x = self.channel_trans(x)
        b,c,h,w = x.shape
        x_sr = self.spatial_reduce(x)
        #x_sr = self.norm(x_sr)
        q = self.q(x).view(b, self.dim, -1).permute(0, 2, 1)
        k = self.k(x_sr).view(b,self.dim,-1)
        v = self.v(x_sr).view(b, self.dim, -1).permute(0, 2, 1)
        attn = torch.bmm(q,k)
        attn = (self.dim ** -.5) * attn
        attn = F.softmax(attn, dim=-1)
        
        attn = torch.bmm(attn, v).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        attn = self.conv_out1(attn)
        
        x = torch.cat([x, attn], dim=1)
        x = self.conv_out2(x)
        x = self.conv_out3(x)
        return x
