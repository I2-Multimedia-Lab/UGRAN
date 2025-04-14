import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import *
import numpy as np
import time
class URA(nn.Module):
    r""" Uncertainty Refinement Attention. 
    
    Args:
        dim (int): Number of low-level feature channels.
        dim1, dim2 (int): Number of high-level feature channels.
        embed_dim (int): Dimension for attention.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    """

    def __init__(self, in_channel, out_channel = 1, dim = 64, base_size = [384,384], stage = None):
        super(URA, self).__init__()

        self.base_size = base_size
        self.ratio = stage
        self.dim = dim
        self.min_size = base_size[0] // 16
        self.max_size = base_size[0] // 4
        self.pthreshold = 0.2
        self.norm = nn.BatchNorm2d(dim)
        self.lnorm = nn.BatchNorm2d(dim)

        self.mha = nn.MultiheadAttention(dim, 1, batch_first = True)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

        self.conv_out1 = nn.Linear(dim, dim)
        self.conv_out3 = Conv2d(dim, dim, 3, relu = True)
        self.conv_out4 = Conv2d(dim, out_channel, 1)

        self.forward = self._ablation

        self.ptime = 0.0 # partition time
        self.rtime = 0.0 # reverse time
        self.etime = 0.0 # execute time

    def initialize(self):
        weight_init(self)
        
    def ADP(self, x, l, umap, p):
        B,C,H,W = x.shape
        h,w = [H//2,W//2]
        st = time.process_time()
        x_w = x.view(B,C,2,h,2,w).permute(2,4,0,1,3,5).contiguous().view(4,B,C,h,w)
        l_w = l.view(B,C,2,h,2,w).permute(2,4,0,1,3,5).contiguous().view(4,B,C,h,w)
        u_w = umap.view(B,1,2,h,2,w).permute(2,4,0,1,3,5).contiguous().view(4,B,1,h,w)
        
        # For partition illustration
        p_w = p.view(B,1,2,h,2,w).permute(2,4,0,1,3,5).contiguous().view(4,B,1,h,w)
        for i in range(0,4):
            p_w[i][0][0][0] = 0.6
            p_w[i][0][0][-1] = 0.6
            p_w[i][0][0][:,0] = 0.6
            p_w[i][0][0][:,-1] = 0.6
        
        et = time.process_time()
        self.ptime+=(et-st)
        
        for i in range(0,4):
            p = torch.sum(u_w[i])/(h*w)
            if (p < self.pthreshold and h > self.min_size) or h > self.max_size: # partition or not
                x_w[i],p_w[i] = self.ADP(x_w[i],l_w[i],u_w[i],p_w[i])
            else:
                st = time.process_time()
                q = x_w[i].flatten(-2).transpose(-1,-2)
                k = l_w[i].flatten(-2).transpose(-1,-2)
                v = l_w[i].flatten(-2).transpose(-1,-2)
                u = u_w[i].flatten(-2).transpose(-1,-2) 
                umask = u @ u.transpose(-1,-2)           
                attn_mask = (umask<1).bool()
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-1e10")) # as negative infinity
                attn,_ = self.mha(q,k,v,attn_mask = new_attn_mask)
                attn = self.conv_out1(attn).transpose(-2,-1).view(B, C, h, w)
                x_w[i] += attn
                et = time.process_time()
                self.etime += (et-st)
                
        st = time.process_time()
        x_w = x_w.permute(1,2,0,3,4).view(B,C,2,2,h,w).permute(0,1,2,4,3,5).reshape(B,C,H,W)
        p_w = p_w.permute(1,2,0,3,4).view(B,1,2,2,h,w).permute(0,1,2,4,3,5).reshape(B,1,H,W)
        et = time.process_time()
        self.rtime+=(et-st)
        return x_w,p_w

    def _forward(self, x, l, umap):
        B,C,H,W = x.shape
        p = torch.ones((B,1,H,W))
        _u=torch.where(umap>0.01,1.0,0.0)

        x,p = self.ADP(x,l,_u,p)
        x = self.conv_out3(x)
        out = self.conv_out4(x)
        
        return x, out, p, self.ptime, self.etime
    
    def _ablation(self, x, l, umap):
        out = self.conv_out4(x)
        return x, out, out, self.ptime,self.etime
    
def window_partition(x, window_size):
    """
    Modified from Swin Transformer ("https://github.com/microsoft/Swin-Transformer")
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, C, window_size, window_size)
    
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
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous().view(B, -1, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, -1, H // window_size, W // window_size, window_size, window_size)
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, -1, H, W)
    """
    return x
