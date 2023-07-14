import torch
import torch.nn as nn
import torch.nn.functional as F
#from Models.modules import *
from .modules import *
import cv2
import numpy as np
import datetime
import time
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
        self.window_size = base_size[0]//8
        self.channel_trans = Conv2d(c_num,depth,1)
        self.pthreshold = 0.2
        self.norm = nn.BatchNorm2d(depth)
        self.lnorm = nn.BatchNorm2d(depth)

        self.mha = nn.MultiheadAttention(depth,1,batch_first=True)
        self.q = nn.Linear(depth,depth)
        self.k = nn.Linear(depth,depth)
        self.v = nn.Linear(depth,depth)

        self.conv_out1 = nn.Linear(depth,depth)
        self.conv_out3 = Conv2d(depth, depth, 3, relu=True)
        self.conv_out4 = Conv2d(depth, out_channel, 1)

        self.forward = self._ablation

        self.ptime = 0.0
        self.rtime = 0.0
        self.etime = 0.0

    def initialize(self):
        weight_init(self)
        
    def DWPA(self, x, l, umap, p):
        B,C,H,W = x.shape
        h,w = [H//2,W//2]
        st = time.process_time()
        x_w = x.view(B,C,2,h,2,w).permute(2,4,0,1,3,5).contiguous().view(4,B,C,h,w)
        l_w = l.view(B,C,2,h,2,w).permute(2,4,0,1,3,5).contiguous().view(4,B,C,h,w)
        u_w = umap.view(B,1,2,h,2,w).permute(2,4,0,1,3,5).contiguous().view(4,B,1,h,w)
        
        p_w = p.view(B,1,2,h,2,w).permute(2,4,0,1,3,5).contiguous().view(4,B,1,h,w)
        for i in range(0,4):
            p_w[i][0][0][0] = 0.6
            p_w[i][0][0][-1] = 0.6
            p_w[i][0][0][:,0] = 0.6
            p_w[i][0][0][:,-1] = 0.6
        
        et = time.process_time()
        self.ptime+=(et-st)
        for i in range(0,4):
            #p = np.random.rand()
            #print(p)
            p = torch.sum(u_w[i])/(h*w)
            #print(p)
            if (p < self.pthreshold and h > 24) or h > 96: # partition or not
                x_w[i],p_w[i] = self.DWPA(x_w[i],l_w[i],u_w[i],p_w[i])
                #x_w[i] = self.DWPA(x_w[i],l_w[i],u_w[i])
            else:
                st = time.process_time()
                q = x_w[i].flatten(-2).transpose(-1,-2)
                k = l_w[i].flatten(-2).transpose(-1,-2)
                v = l_w[i].flatten(-2).transpose(-1,-2)
                u = u_w[i].flatten(-2).transpose(-1,-2) 
                umask = u @ u.transpose(-1,-2)           
                attn_mask = (umask<1).bool()
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-1e10"))
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


    
    def _forward(self,x,l,umap,ratio=1):
                
        B,C,H,W = x.shape
        #umap = self.get_uncertain(smap,(H,W))
        p = torch.ones((B,1,H,W))
        #print(torch.sum(umap)/(H*W))
        #_u = (umap>0).bool()
        _u=torch.where(umap>0.01,1.0,0.0)
        #print(torch.sum(_u)/(H*W))

        x,p = self.DWPA(x,l,_u,p)
        if(ratio != 1):
            x = F.interpolate(x, [H*ratio,W*ratio], mode='bilinear', align_corners=True)
        x = self.conv_out3(x)
        out = self.conv_out4(x)
        if(ratio != 1):
            x = F.interpolate(x, [H,W], mode='bilinear', align_corners=True)
        return x, out, p
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
