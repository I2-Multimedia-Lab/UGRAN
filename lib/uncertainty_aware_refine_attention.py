import torch
import torch.nn as nn
import torch.nn.functional as F
#from Models.modules import *
from .modules import *
import cv2
import numpy as np
class URA(nn.Module):
    # Window-based Context Attention
    def __init__(self, in_channel, out_channel=1, depth=64, base_size=[384,384], window_size = 12, c_num=3, stage=None):
        super(URA, self).__init__()
        self.base_size=base_size
        if base_size is not None and stage is not None:
            self.stage_size = (base_size[0] // (2 ** stage), base_size[1] // (2 ** stage))
        else:
            self.stage_size = None
        self.ratio = stage
        self.depth = depth
        self.depth = depth
        self.window_size = window_size
        self.channel_trans = Conv2d(c_num,depth,1)
        self.threshold = 0.5
        #self.lthreshold = nn.Parameter(torch.tensor([0.5]))

        k = cv2.getGaussianKernel(7, 1)
        k = np.outer(k, k)
        k = torch.tensor(k).float()
        self.kernel = k.repeat(1, 1, 1, 1).cuda()


        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), 1))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Sequential(
            nn.Linear(depth,depth)
        )
        self.k = nn.Sequential(
            nn.Linear(depth,depth)
        )
        self.v = nn.Sequential(
            nn.Linear(depth,depth)
        )

        self.conv_out1 = Conv2d(depth,depth,3,relu=True)
        
        #self.conv_out2 = Conv2d(in_channel+depth, depth, 3, relu=True)
        self.conv_out3 = Conv2d(depth, depth, 3, relu=True)
        self.conv_out4 = Conv2d(depth, out_channel, 1)

        self.forward = self._forward

    def get_uncertain(self,smap,shape):
        smap = F.interpolate(smap, size=shape, mode='bilinear', align_corners=False)
        smap = torch.sigmoid(smap)
        p = smap - self.threshold
        cg = self.threshold - torch.abs(p)
        cg = F.pad(cg, (7 // 2, ) * 4, mode='reflect')
        cg = F.conv2d(cg, self.kernel * 4, groups=1)
        return cg/cg.max()

    def initialize(self):
        weight_init(self)
        
    def _forward(self, x, l, map_s,map_l=None):

        B,C,H,W = x.shape
        cg = self.get_uncertain(map_s,(H,W))

        x_uncertain = l
        cg_ = F.interpolate(cg,size=l.shape[-2:])
        
        mask = cg.reshape(B,1,-1).transpose(1,2) @ cg_.reshape(B,1,-1)
        mask = torch.log(mask+0.5)
        x_ = x.reshape(B,C,-1).transpose(1,2)
        x_uncertain = x_uncertain.reshape(B, C, -1).permute(0, 2, 1)
        q = self.q(x_)
        k = self.k(x_uncertain)
        v = self.v(x_uncertain)
        attn = q @ k.transpose(-2, -1)
        attn = (self.depth ** -.5) * attn

        attn = F.softmax(attn+mask, dim=-1)
        
        attn = (attn @ v).transpose(1,2).reshape(B,C,H,W)
        attn = self.conv_out1(attn)
        x = x+attn
        
        #x = self.conv_out2(x)
        x = self.conv_out3(x)
        out = self.conv_out4(x)

        return x, out, cg
    
    def _ablation(self,x, map_s,map_l=None):
        out = self.conv_out4(x)
        return x, out, out
    
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
