import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.modules import *
from .layers import *
class WCA(nn.Module):
    # Window-based Context Attention
    def __init__(self, in_channel, dim, out_channel=1, depth=64, base_size=[384,384], window_size = 12, stage=None):
        super(WCA, self).__init__()
        
        if base_size is not None and stage is not None:
            self.stage_size = (base_size[0] // (2 ** stage), base_size[1] // (2 ** stage))
        else:
            self.stage_size = None
        self.ratio = stage
        self.depth = depth
        self.dim = dim
        self.window_size = window_size
        self.spatial_reduce = Conv2d(dim,dim,self.ratio,self.ratio)
        self.norm = nn.BatchNorm2d(in_channel)
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
        self.conv_out4 = Conv2d(depth, out_channel, 1)
    def forward(self, x_in, context):
        x = self.norm(x_in)
        x_windows = window_partition(x,self.window_size)
        c_windows = window_partition(context,self.window_size)
        b = x_windows.shape[0]
        q = self.q(x_windows).view(b, self.dim, -1).permute(0, 2, 1)
        k = self.k(c_windows).view(b,self.dim,-1)
        v = self.v(c_windows).view(b, self.dim, -1).permute(0, 2, 1)
        attn = torch.bmm(q,k)
        attn = (self.dim ** -.5) * attn
        attn = F.softmax(attn, dim=-1)
        
        attn = torch.bmm(attn, v).permute(0, 2, 1).contiguous().view(b, -1, self.window_size, self.window_size)
        x_reverse = window_reverse(attn,self.window_size)
        x_cat = torch.cat([x_in,x_reverse],dim=1)
        x = self.conv_out2(x_cat)
        x = self.conv_out3(x)
        out = self.couv_out4(x)
        return x, out
    def window_partition(x, window_size):
        """
        Args:
            x: (B, H, W, C)
            window_size (int): window size

        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        """
        B, C, H, W = x.shape
        x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
        windows = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1, C, window_size, window_size)
        return windows


    def window_reverse(windows, window_size, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): Window size
            H (int): Height of image
            W (int): Width of image

        Returns:
            x: (B, H, W, C)
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, -1, H // window_size, W // window_size, window_size, window_size)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1, H, W)
        return x
