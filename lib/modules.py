import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def initialize(self):
        weight_init(self)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    def flops(self,N):
        return N*(self.in_features+self.out_features)*self.hidden_features

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def initialize(self):
        weight_init(self)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    
    def flops(self,N):
        flops = 0
        #q
        flops += N*self.dim*self.dim*3
        #qk
        flops += self.num_heads*N*self.dim//self.num_heads*N
        #att v
        flops += self.num_heads*N*self.dim//self.num_heads*N
        #proj
        flops += N*self.dim*self.dim
        return flops

class CrossAttention(nn.Module):
    def __init__(self, dim1,dim2, dim, out_channel=None, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim
        self.dim1 = dim1
        self.dim2 = dim2
        self.scale = qk_scale or head_dim ** -0.5
        self.out_channel = out_channel or dim1

        self.q1 = nn.Linear(dim1, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, self.out_channel)

        self.k2 = nn.Linear(dim2, dim, bias=qkv_bias)
        self.v2 = nn.Linear(dim2, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
    def initialize(self):
        weight_init(self)
    def forward(self, fea, depth_fea):
        _, N1, _ = fea.shape
        B, N, _ = depth_fea.shape
        C = self.dim
        q1 = self.q1(fea).reshape(B, N1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        # q [B, nhead, N, C//nhead]

        k2 = self.k2(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v2 = self.v2(depth_fea).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        fea = (attn @ v2).transpose(1, 2).reshape(B, N1, C)
        fea = self.proj(fea)
        fea = self.proj_drop(fea)

        return fea
    
    def flops(self,N1,N2):
        flops = 0
        #q
        flops += N1*self.dim1*self.dim
        #kv
        flops += N2*self.dim2*self.dim*2
        #qk
        flops += self.num_heads*N1*self.dim//self.num_heads*N2
        #att v
        flops += self.num_heads*N1*self.dim//self.num_heads*N2
        #proj
        flops += N1*self.dim*self.dim1
        return flops

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, (nn.Conv2d,nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            #nn.init.xavier_normal_(m.weight, gain=1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d,nn.LayerNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Sequential,nn.ModuleList)):
            weight_init(m)
        elif isinstance(m, (nn.Conv1d, nn.ReLU, nn.GELU, nn.Sigmoid, nn.PReLU, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity, nn.Upsample, nn.Dropout, nn.MultiheadAttention, nn.MaxPool2d)):
            pass
        else:
            m.initialize()

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, padding='same', bias=False, bn=True, relu=False):
        super(Conv2d, self).__init__()
        if '__iter__' not in dir(kernel_size):
            kernel_size = (kernel_size, kernel_size)
        if '__iter__' not in dir(stride):
            stride = (stride, stride)
        if '__iter__' not in dir(dilation):
            dilation = (dilation, dilation)

        if padding == 'same':
            width_pad_size = kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1)
            height_pad_size = kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1)
        elif padding == 'valid':
            width_pad_size = 0
            height_pad_size = 0
        else:
            if '__iter__' in dir(padding):
                width_pad_size = padding[0] * 2
                height_pad_size = padding[1] * 2
            else:
                width_pad_size = padding * 2
                height_pad_size = padding * 2

        width_pad_size = width_pad_size // 2 + (width_pad_size % 2 - 1)
        height_pad_size = height_pad_size // 2 + (height_pad_size % 2 - 1)
        pad_size = (width_pad_size, height_pad_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad_size, dilation, groups, bias=bias)
        self.initialize()

        if bn is True:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        
        if relu is True:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
    def initialize(self):
        weight_init(self)

class SE(nn.Module):
    def __init__(self, dim, r=16):
        super(SE, self).__init__()
        self.se = nn.Sequential(
            nn.Linear(dim,dim//r),
            nn.GELU(),
            nn.Linear(dim//r,dim)
        )
        
    def initialize(self):
        weight_init(self)

    def forward(self, fea):
        b, c, _, _ = fea.size()
        y = F.adaptive_avg_pool2d(fea,1)
        y = y.view(b, c)
        y = self.se(y)
        y = y.view(b, c, 1, 1)

        return fea*y

class GCT(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super(GCT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1, 1))
        self.eps = eps

    def initialize(self):
        weight_init(self)

    def forward(self, x):
        embedding = (x.pow(2).sum((2,3), keepdim=True) + self.eps).pow(0.5) * self.alpha
        norm = self.gamma / (embedding.pow(2).mean(dim=1, keepdim=True) + self.eps).pow(0.5)
        gate = 1. + torch.tanh(embedding * norm + self.beta)

        return x * gate

class ICE(nn.Module):
# --------------------------------------------------------
# Integrity Channel Enhancement
# Copyright (c) 2020 mczhuge
# Licensed under The MIT License 
# https://github.com/mczhuge/ICON
# --------------------------------------------------------

    def __init__(self, num_channels=64, ratio=8):
        super(ICE, self).__init__()
        self.conv_cross = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, bias=False)    
        self.bn_cross = nn.BatchNorm2d(num_channels)
        self.eps = 1e-5   
        self.conv_mask = nn.Conv2d(num_channels, 1, kernel_size=1)#context Modeling
        self.softmax = nn.Softmax(dim=2)
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // ratio, kernel_size=1),
            nn.LayerNorm([num_channels // ratio, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels // ratio, num_channels, kernel_size=1)
        )
        
    def initialize(self):
        weight_init(self)

    def forward(self, x):
        x = F.relu(self.bn_cross(self.conv_cross(x))) #[B, C, H, W]
        context = (x.pow(2).sum((2,3), keepdim=True) + self.eps).pow(0.5) # [B, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)

        return x * channel_add_term

class ImagePyramid:
# --------------------------------------------------------
# InSPyReNet
# Copyright (c) 2021 Taehun Kim
# Licensed under The MIT License 
# https://github.com/plemeri/InSPyReNet
# --------------------------------------------------------

    def __init__(self, ksize=7, sigma=1, channels=1):
        self.ksize = ksize
        self.sigma = sigma
        self.channels = channels
        self.uthreshold = 0.5
        k = cv2.getGaussianKernel(ksize, sigma)
        k = np.outer(k, k)
        k = torch.tensor(k).float()
        self.kernel = k.repeat(channels, 1, 1, 1)
        
    def to(self, device):
        self.kernel = self.kernel.to(device)
        return self
        
    def cuda(self, idx=None):
        if idx is None:
            idx = torch.cuda.current_device()
            
        self.to(device="cuda:{}".format(idx))
        return self

    def expand(self, x):
        z = torch.zeros_like(x)
        x = torch.cat([x, z, z, z], dim=1)
        x = F.pixel_shuffle(x, 2)
        x = F.pad(x, (self.ksize // 2, ) * 4, mode='reflect')
        x = F.conv2d(x, self.kernel * 4, groups=self.channels)
        return x

    def reduce(self, x):
        x = F.pad(x, (self.ksize // 2, ) * 4, mode='reflect')
        x = F.conv2d(x, self.kernel, groups=self.channels)
        x = x[:, :, ::2, ::2]
        return x

    def deconstruct(self, x):
        reduced_x = self.reduce(x)
        expanded_reduced_x = self.expand(reduced_x)

        if x.shape != expanded_reduced_x.shape:
            expanded_reduced_x = F.interpolate(expanded_reduced_x, x.shape[-2:])

        laplacian_x = x - expanded_reduced_x
        return reduced_x, laplacian_x

    def reconstruct(self, x, laplacian_x):
        expanded_x = self.expand(x)
        if laplacian_x.shape != expanded_x:
            laplacian_x = F.interpolate(laplacian_x, expanded_x.shape[-2:], mode='bilinear', align_corners=True)
  
        return expanded_x + laplacian_x
    
    def get_uncertain(self,smap,shape):
        smap = F.interpolate(smap, size=shape, mode='bilinear', align_corners=False)
        smap = torch.sigmoid(smap)
        p = smap - self.uthreshold
        cg = self.uthreshold - torch.abs(p)
        cg = F.pad(cg, (self.ksize // 2, ) * 4, 'constant', 0)
        cg = F.conv2d(cg, self.kernel * 4, groups=1)
        return cg/cg.max()
