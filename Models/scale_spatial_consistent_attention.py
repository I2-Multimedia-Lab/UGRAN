import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.modules import *
from .layers import *
class SSCA(nn.Module):
    def __init__(self, in_channel, depth=64, base_size=[384,384], stage=1):
        super(SSCA, self).__init__()
        
        if base_size is not None and stage is not None:
            self.stage_size = (base_size[0] // (2 ** stage), base_size[1] // (2 ** stage))
        else:
            self.stage_size = None
        self.ratio = stage
        self.depth = depth
        '''
        self.spatial_reduce = Conv2d(depth,depth,self.ratio,self.ratio)
        self.norm = nn.LayerNorm(depth)
        self.channel_reduce = Conv2d(in_channel,depth,3,relu=True)
        self.q = nn.Sequential(
            Conv2d(depth,depth,3,relu=True),
        )
        self.k = nn.Sequential(
            Conv2d(depth,depth,1,relu=True),
        )
        self.v = nn.Sequential(
            Conv2d(depth,depth,3,relu=True),
        )

        self.conv_out1 = Conv2d(depth,depth,3,relu=True)
        self.conv_out2 = Conv2d(in_channel+depth, depth, 3, relu=True)
        self.conv_out3 = Conv2d(depth, depth, 3, relu=True)
        '''
        self.conv = Conv2d(in_channel,depth,3)
    def forward(self, x_i):
        '''
        x = self.channel_reduce(x_i)
        b,c,h,w = x.size
        x_sr = self.spatial_reduce(x)
        q = self.q(x).view(b, self.depth, -1).permute(0, 2, 1)
        k = self.k(x_sr).view(b,self.depth,-1)
        v = self.v(x_sr).view(b, self.depth, -1).permute(0, 2, 1)
        attn = torch.bmm(q,k)
        attn = (self.depth ** -.5) * attn
        attn = F.softmax(attn, dim=-1)
        
        attn = torch.bmm(attn, v).permute(0, 2, 1).contiguous().view(b, -1, h, w)
        attn = self.conv_out1(attn)
        
        x = torch.cat([x_i, attn], dim=1)
        x = self.conv_out2(x)
        x = self.conv_out3(x)
        '''
        x = self.conv(x_i)
        return x
