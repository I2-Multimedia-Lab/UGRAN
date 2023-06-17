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
        self.threshold = 0.5
        #self.lthreshold = nn.Parameter(torch.tensor([0.5]))
        self.norm = nn.BatchNorm2d(depth)
        self.lnorm = nn.BatchNorm2d(depth)
        
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

        self.forward = self.__forward

        self.ptime = 0.0
        self.rtime = 0.0
        self.etime = 0.0

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
        
    '''
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(-1, C, window_size, window_size)    
    '''
    def DWPA(self, x, l, umap):
        B,C,H,W = x.shape
        h,w = [H//2,W//2]
        st = time.process_time()
        x_w = x.view(B,C,2,h,2,w).permute(2,4,0,1,3,5).contiguous().view(4,B,C,h,w)
        l_w = l.view(B,C,2,h,2,w).permute(2,4,0,1,3,5).contiguous().view(4,B,C,h,w)
        u_w = umap.view(B,1,2,h,2,w).permute(2,4,0,1,3,5).contiguous().view(4,B,1,h,w)
        et = time.process_time()
        self.ptime+=(et-st)
        for i in range(0,4):
            #p = np.random.rand()
            #print(p)
            if h > 48: # partition or not
                self.DWPA(x_w[i],l_w[i],u_w[i])
            else:
                #x_w = x_w.view(-1,C,h,w)
                #l_w = l_w.view(-1,C,h,w)
                #u_w = u_w.view(-1,C,h,w)
                st = time.process_time()
                q= self.q(x_w[i].flatten(-2).transpose(-1,-2))
                k = self.k(l_w[i].flatten(-2).transpose(-1,-2))
                v = self.v(l_w[i].flatten(-2).transpose(-1,-2))
                attn = q @ k.transpose(-2,-1)
                attn = (self.depth ** -.5) * attn
                attn = (attn @ v).transpose(-2,-1).view(B, C, h, w)
                attn = self.conv_out1(attn)
                x_w[i] += attn
                et = time.process_time()
                self.etime += (et-st)
        st = time.process_time()
        x_w = x_w.permute(1,2,0,3,4).view(B,C,2,2,h,w).permute(0,1,2,4,3,5).reshape(B,C,H,W)
        et = time.process_time()
        self.rtime+=(et-st)
        #return x_w

    def DWPA_(self, x, l, umap):
        
        B,C,H,W = x.shape
        #print(x.shape)
        #print()
        h,w = [H//2,W//2]
        st = time.process_time()
        x_w = x.view(B,C,2,h,2,w).permute(2,4,0,1,3,5).contiguous().view(4,B,C,h,w)
        l_w = l.view(B,C,2,h,2,w).permute(2,4,0,1,3,5).contiguous().view(4,B,C,h,w)
        u_w = umap.view(B,1,2,h,2,w).permute(2,4,0,1,3,5).contiguous().view(4,B,1,h,w)
        et = time.process_time()
        self.ptime+=(et-st)
        plistx = []
        plistl = []
        plistu = []
        pid = []

        elistx = []
        elistl = []
        elistu = []
        eid= []

        for i in range(0,4):
            #p = np.random.rand()
            #print(p)
            if h > 12: # partition or not
                #self.DWPA(x_w[i],l_w[i],u_w[i])
                #plistx.append(x_w[i])
                #plistl.append(l_w[i])
                #plistu.append(u_w[i])
                pid.append(i)
            else:
                '''#x_w = x_w.view(-1,C,h,w)
                #l_w = l_w.view(-1,C,h,w)
                #u_w = u_w.view(-1,C,h,w)
                st = time.process_time()
                q= self.q(x_w[i].flatten(-2).transpose(-1,-2))
                k = self.k(l_w[i].flatten(-2).transpose(-1,-2))
                v = self.v(l_w[i].flatten(-2).transpose(-1,-2))
                attn = q @ k.transpose(-2,-1)
                attn = (self.depth ** -.5) * attn
                attn = (attn @ v).transpose(-2,-1).view(B, C, h, w)
                attn = self.conv_out1(attn)
                x_w[i] += attn
                et = time.process_time()
                self.etime += (et-st)'''
                #elistx.append(x_w[i])
                #elistl.append(l_w[i])
                #elistu.append(u_w[i])
                eid.append(i)
                #print(id(x_w[i]),id(elistx[-1]))
        #print(id(x_w[0]),id(x_w[-1]),id(x_w))
        le = len(eid)
        lp = len(pid)
        #print(le,lp)
        x_ = torch.split(x_w,1,0)
        l_ = torch.split(l_w,1,0)
        u_ = torch.split(u_w,1,0)
        if(le>0):
            # execute current window
            #elistx = torch.cat(elistx,dim=0)
            #elistl = torch.cat(elistl,dim=0)
            #elistu = torch.cat(elistu,dim=0)
            ex = torch.cat([x_[i] for i in eid],dim=0)
            el = torch.cat([l_[i] for i in eid],dim=0)
            #x_w = x_w.permute(eid[0],eid[1],eid[2],eid[3])
   
            q = self.q(ex.flatten(-2).transpose(-1,-2))
            k = self.k(el.flatten(-2).transpose(-1,-2))
            v = self.v(el.flatten(-2).transpose(-1,-2))
            attn = q @ k.transpose(-2,-1)
            attn = (self.depth ** -.5) * attn
            attn = (attn @ v).transpose(-2,-1).view(-1, C, h, w)
            attn = self.conv_out1(attn).view(le,-1,C,h,w)
            ex += attn
            for i in range(0,le):
                x_w[eid[i]] = ex[i]
            
            #for i in eid:
            #    print(i)
            #print(x_w[eid[0]].max(),elistx[0].max())
            #print(x_w[eid[0]].min(),elistx[0].min())
                

        if(lp>0):
            # execute smaller window
            #plistx = torch.cat(plistx,dim=0)
            #plistl = torch.cat(plistl,dim=0)
            #plistu = torch.cat(plistu,dim=0)

            px = torch.cat([x_[i] for i in pid],dim=0).view(-1,C,h,w)
            pl = torch.cat([l_[i] for i in pid],dim=0).view(-1,C,h,w)
            pu = torch.cat([u_[i] for i in pid],dim=0).view(-1,1,h,w)
            px = self.DWPA_(px,pl,pu)
            px = px.view(lp,-1,C,h,w)
            for i in range(0,len(pid)):
                x_w[pid[i]] = px[i]
            #for i in pid:
            #    print(i)
            #print(x_w[pid[0]].max(),plistx[0].max())
            #print(x_w[pid[0]].min(),plistx[0].min())
                
        st = time.process_time()
        #print(x_w.shape)
        x_w = x_w.permute(1,2,0,3,4).view(B,C,2,2,h,w).permute(0,1,2,4,3,5).reshape(B,C,H,W)
        et = time.process_time()
        self.rtime+=(et-st)
        return x_w

    def _forward(self, x, l, map_s,map_l=None):
        
        B,C,H,W = x.shape
        cg = self.get_uncertain(map_s,(H,W))

        x_uncertain = l
        
        st = time.process_time()
        #cg_windows = window_partition(cg,self.window_size).flatten(2).transpose(1,2)
        x_windows = window_partition(self.norm(x),self.window_size).flatten(2).transpose(1,2)
        c_windows = window_partition(self.norm(x_uncertain),self.window_size).flatten(2).transpose(1,2)
        et = time.process_time()
        self.ptime+=(et-st)
        st = time.process_time()
        b = x_windows.shape[0]
        q = self.q(x_windows)
        k = self.k(c_windows)
        v = self.v(c_windows)
        attn = q @ k.transpose(-2, -1)
        attn = (self.depth ** -.5) * attn

        #cg_ = cg_windows @ cg_windows.transpose(1,2)+0.5
        #attn=attn+torch.log(cg_)
        #relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
        #    self.window_size * self.window_size, self.window_size * self.window_size, -1)  # Wh*Ww,Wh*Ww,nH
        #relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        #attn = attn + relative_position_bias#.unsqueeze(0)

        attn = F.softmax(attn, dim=-1)
        attn = (attn @ v).view(b, self.window_size, self.window_size, C)
        et = time.process_time()
        self.etime+=(et-st)        
        st = time.process_time()
        x_reverse = window_reverse(attn,self.window_size,H,W)
        et = time.process_time()
        self.rtime+=(et-st)
        x_reverse = self.conv_out1(x_reverse)
        x = x+x_reverse
        
        #x = self.conv_out2(x)
        x = self.conv_out3(x)
        out = self.conv_out4(x)

        return x, out, cg
    
    def __forward(self,x,l,map_s,map_l=None):
                
        B,C,H,W = x.shape
        cg = self.get_uncertain(map_s,(H,W))

        x = self.DWPA_(x,l.detach(),cg.detach())
        
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
