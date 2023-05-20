import torch
import torch.nn as nn
from timm.models.layers import DropPath
from .modules import *
class MIA(nn.Module):
    r""" Multilevel Interaction Block. 
    
    Args:
        dim (int): Number of low-level feature channels.
        dim1, dim2 (int): Number of high-level feature channels.
        embed_dim (int): Dimension for attention.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    """
    
    def __init__(self,in_channel,out_channel,dim1=None,dim2=None,dim3=None,embed_dim = 384,num_heads = 6,mlp_ratio = 3., qkv_bias = False, qk_scale = None,drop = 0.,attn_drop = 0.,drop_path = 0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(MIA, self).__init__()
        self.dim = in_channel
        self.mlp_ratio = mlp_ratio
        self.embed_dim = embed_dim

        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3

        self.norm0 = norm_layer(in_channel)
        self.ca = SE(dim=self.dim)
        self.ct = Conv2d(self.dim,self.embed_dim,1)
        if self.dim1:
            #self.ca1 = SE(dim=dim1)
            self.interact1 = CrossAttention(dim1 = in_channel,dim2 = dim1,dim = embed_dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,attn_drop=attn_drop,proj_drop=drop)
            self.norm1 = norm_layer(dim1)

            self.norm = nn.BatchNorm2d(in_channel)
            self.mlp = nn.Sequential(
                nn.Conv2d(in_channel, in_channel*mlp_ratio,1),
                act_layer(),
                nn.Conv2d(in_channel*mlp_ratio, in_channel,1),
            )
        if self.dim2:
            #self.ca2 = SE(dim=dim2)
            self.interact2 = CrossAttention(dim1 = embed_dim,dim2 = dim2,dim = embed_dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,attn_drop=attn_drop,proj_drop=drop)
            self.norm2 = norm_layer(dim2)
        if self.dim3:
            #self.ca2 = SE(dim=dim2)
            self.interact3 = CrossAttention(dim1 = embed_dim,dim2 = dim3,dim = embed_dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,attn_drop=attn_drop,proj_drop=drop)
            self.norm3 = norm_layer(dim3)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.proj = Conv2d(in_channel,out_channel,1)
        
        self.forward = self._forward
        if self.forward == self._ablation:
            self.res = Conv2d(in_channel,out_channel,1)
    def initialize(self):
        weight_init(self)

    def _forward(self,fea,fea_1=None,fea_2=None,fea_3=None):
        fea = self.ca(fea)
        #fea = self.ct(fea)
        if self.dim1!=None and fea_1!=None:
            B,C,H,W = fea.shape
            fea = fea.reshape(B,C,-1).transpose(1,2)
            fea = self.norm0(fea)
            fea_1 = fea_1.reshape(B,self.dim1,-1).transpose(1,2)
            fea_1 = self.norm1(fea_1)
            fea_1 = self.interact1(fea,fea_1)
                
            if self.dim2!=None and fea_2!=None:
                fea_2 = fea_2.reshape(B,self.dim2,-1).transpose(1,2)
                fea_2 = self.norm2(fea_2)
                fea_2 = self.interact2(fea,fea_2)
            if self.dim3!=None and fea_3!=None:
                fea_3 = fea_3.reshape(B,self.dim3,-1).transpose(1,2)
                fea_3 = self.norm3(fea_3)
                fea_3 = self.interact3(fea,fea_3)
            fea = fea + fea_1 
            if self.dim2!=None and fea_2!=None:
                fea = fea + fea_2
            if self.dim3!=None and fea_3!=None:
                fea = fea + fea_3
            fea = fea.transpose(1,2).reshape(B,C,H,W)
            #fea = fea + self.drop_path(self.mlp(self.norm(fea)))
        fea = self.proj(fea)
        return fea
    
    def _ablation(self,fea,fea_1=None,fea_2=None,fea_3=None):
        fea = self.res(fea)
        return fea

    def flops(self,N1,N2,N3=None):
        flops = 0
        flops += self.interact1.flops(N1,N2)
        if N3:
            flops += self.interact2.flops(N1,N3)
        flops += self.dim*N1
        flops += 2*N1*self.dim*self.dim*self.mlp_ratio
        return flops

class MIA1(nn.Module):
    r""" Multilevel Interaction Block. 
    
    Args:
        dim (int): Number of low-level feature channels.
        dim1, dim2 (int): Number of high-level feature channels.
        embed_dim (int): Dimension for attention.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    """
    
    def __init__(self,in_channel,out_channel,dim1=None,dim2=None,dim3=None,embed_dim = 384,num_heads = 6,mlp_ratio = 3., qkv_bias = False, qk_scale = None,drop = 0.,attn_drop = 0.,drop_path = 0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(MIA, self).__init__()
        self.with_channel = False
        self.after_relu = False
        
        mid_channel=in_channel
        if dim1 != None:
            mid_channel = dim1
            #self.ct = nn.Conv2d(in_channel,mid_channel,1)
            self.f_x = nn.Sequential(
                                nn.Conv2d(in_channel, in_channel, 
                                          kernel_size=1, bias=False),
                                nn.BatchNorm2d(in_channel)
                                )
            self.f_y = nn.Sequential(
                                nn.Conv2d(dim1, in_channel, 
                                          kernel_size=1, bias=False),
                                nn.BatchNorm2d(in_channel)
                                )
        self.proj = nn.Conv2d(in_channel,out_channel,1)
        if self.with_channel:
            self.up = nn.Sequential(
                                    nn.Conv2d(mid_channel, in_channel, 
                                              kernel_size=1, bias=False),
                                    nn.BatchNorm2d(in_channel)
                                   )
        if self.after_relu:
            self.relu = nn.ReLU(inplace=True)
        self.forward = self._ablation
        if self.forward == self._ablation:
            self.res = Conv2d(in_channel,out_channel,1)
    def initialize(self):
        weight_init(self)

    def _forward(self,x,fea_1=None,fea_2=None,fea_3=None):
        y = fea_1
        input_size = x.size()
        if y!=None:
            #x = self.ct(x)
            if self.after_relu:
                y = self.relu(y)
                x = self.relu(x)
            
            y_q = self.f_y(y)
            y_q = F.interpolate(y_q, size=[input_size[2], input_size[3]],
                                mode='bilinear', align_corners=False)
            x_k = self.f_x(x)
            
            if self.with_channel:
                sim_map = torch.sigmoid(self.up(x_k * y_q))
            else:
                sim_map = torch.sigmoid(torch.sum(x_k * y_q, dim=1).unsqueeze(1))
            
            y = F.interpolate(y, size=[input_size[2], input_size[3]],
                                mode='bilinear', align_corners=False)
            x = (1+sim_map)*x# + sim_map*y
        x = self.proj(x)
        return x
    
    def _ablation(self,fea,fea_1=None,fea_2=None,fea_3=None):
        fea = self.res(fea)
        return fea

    def flops(self,N1,N2,N3=None):
        flops = 0
        flops += self.interact1.flops(N1,N2)
        if N3:
            flops += self.interact2.flops(N1,N3)
        flops += self.dim*N1
        flops += 2*N1*self.dim*self.dim*self.mlp_ratio
        return flops
        
if __name__ == '__main__':
    # Test
    model = MIA(dim1=96,dim2=192,dim3=384)
    model.cuda()
    f = []
    f.append(torch.randn((1,3136,96)).cuda())
    f.append(torch.randn((1,784,192)).cuda())
    f.append(torch.randn((1,196,384)).cuda())
    y = model(f[0],f[1],f[2])
    print(y.shape)

