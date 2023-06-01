import torch
import torch.nn as nn
import torch.nn.functional as F
#from Models.modules import *
from .modules import *
class URA(nn.Module):
    # Window-based Context Attention
    def __init__(self, in_channel, out_channel=1, depth=64, base_size=[384,384], window_size = 12, c_num=3, stage=None):
        super(URA, self).__init__()
        
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

    def initialize(self):
        weight_init(self)
        
    def _forward(self, x, l, map_s,map_l=None):
        
        H,W  = x.shape[-2:]
        map_s = F.interpolate(map_s, size=x.shape[-2:], mode='bilinear', align_corners=False)
        map_s = torch.sigmoid(map_s)
        p = map_s - self.threshold
        #fg = torch.clip(p, 0, 1) # foreground
        #bg = torch.clip(-p, 0, 1) # background
        cg = self.threshold - torch.abs(p) # confusion area

        x_uncertain = l*cg
        
        x_windows = window_partition(x,self.window_size).flatten(2).transpose(1,2)
        c_windows = window_partition(x_uncertain,self.window_size).flatten(2).transpose(1,2)
        b = x_windows.shape[0]
        q = self.q(x_windows)
        k = self.k(c_windows)
        v = self.v(c_windows)
        attn = q @ k.transpose(-2, -1)
        attn = (self.depth ** -.5) * attn

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias#.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        
        attn = (attn @ v).view(b, -1, self.window_size, self.window_size)
        x_reverse = window_reverse(attn,self.window_size,H,W)
        x_reverse = self.conv_out1(x_reverse)
        x = x+x_reverse
        
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
