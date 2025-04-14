import torch
import torch.nn as nn
from timm.models.layers import DropPath
from .modules import *
class MIA(nn.Module):
    r""" Multilevel Interaction Attention. 
    
    Args:
        in_channel (int): Number of low-level feature channels. 
        out_channel (int): Number of output feature channels. 
        dim1, dim2 (optional)(int): Number of high-level feature channels. 
        embed_dim (int): Dimension for attention. 
    """

    def __init__(self,in_channel,out_channel,dim1=None,dim2=None,dim3=None,embed_dim = 384,num_heads = 6,mlp_ratio = 3., qkv_bias = False, qk_scale = None,drop = 0.,attn_drop = 0.,drop_path = 0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(MIA, self).__init__()
        self.dim = in_channel
        self.mlp_ratio = mlp_ratio
        self.embed_dim = embed_dim

        self.dim1 = dim1
        self.dim2 = dim2
        self.dim3 = dim3

        self.norm0 = norm_layer(self.embed_dim)
        self.ca = SE(dim = self.dim)
        self.ct = Conv2d(self.dim, self.embed_dim,1)

        if self.dim1:
            #self.ca1 = SE(dim=dim1)
            self.interact1 = nn.Sequential(
                Conv2d(self.embed_dim+self.dim1,self.embed_dim,3,1,1,relu=True),
                Conv2d(self.embed_dim,self.embed_dim,3,1,1,relu=False),
            )
            #CrossAttention(dim1 = self.embed_dim,dim2 = dim1,dim = self.embed_dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,attn_drop=attn_drop,proj_drop=drop)
            self.norm1 = norm_layer(dim1)
        if self.dim2:
            #self.ca2 = SE(dim=dim2)
            self.interact2 = nn.Sequential(
                Conv2d(self.embed_dim+self.dim2,self.embed_dim,3,1,1,relu=True),
                Conv2d(self.embed_dim,self.embed_dim,3,1,1,relu=False),
            )
            #CrossAttention(dim1 = self.embed_dim,dim2 = dim2,dim = self.embed_dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,attn_drop=attn_drop,proj_drop=drop)
            self.norm2 = norm_layer(dim2)
        if self.dim3:
            #self.ca2 = SE(dim=dim2)
            self.interact3 = CrossAttention(dim1 = self.embed_dim,dim2 = dim3,dim = self.embed_dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,attn_drop=attn_drop,proj_drop=drop)
            self.norm3 = norm_layer(dim3)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.proj = nn.Sequential(
            Conv2d(self.embed_dim,self.embed_dim,3),
            Conv2d(self.embed_dim,out_channel,1)
        )        
        self.forward = self._forward
        if self.forward == self._ablation:
            self.res = Conv2d(in_channel,out_channel,1)

    def initialize(self):
        weight_init(self)

    def _forward(self,fea,fea_1=None,fea_2=None,fea_3=None):
        fea = self.ca(fea)
        fea = self.ct(fea)
        if self.dim1!=None and fea_1!=None:
            B,_,H,W = fea.shape
            #fea = fea.reshape(B,self.embed_dim,-1).transpose(1,2)
            #fea = self.norm0(fea)
            _,C,_,_ = fea_1.shape
            #fea_1 = fea_1.reshape(B,C,-1).transpose(1,2)
            #fea_1 = self.norm1(fea_1)
            fea_1 = F.interpolate(fea_1, fea.shape[2:], mode = 'bilinear', align_corners = False)
            fea_1 = self.interact1(torch.concat([fea,fea_1],dim=1))
                
            if self.dim2!=None and fea_2!=None:
                #_,C,_,_ = fea_2.shape
                #fea_2 = fea_2.reshape(B,C,-1).transpose(1,2)
                #fea_2 = self.norm2(fea_2)
                fea_2 = F.interpolate(fea_2, fea.shape[2:], mode = 'bilinear', align_corners = False)
                fea_2 = self.interact1(torch.concat([fea,fea_2],dim=1))
            if self.dim3!=None and fea_3!=None:
                fea_3 = fea_3.reshape(B,self.dim3,-1).transpose(1,2)
                fea_3 = self.norm3(fea_3)
                fea_3 = self.interact3(fea,fea_3)
            fea = fea + fea_1 
            if self.dim2!=None and fea_2!=None:
                fea = fea + fea_2
            if self.dim3!=None and fea_3!=None:
                fea = fea + fea_3
            #fea = fea + self.drop_path(self.mlp(self.norm(fea)))
            #fea = fea.transpose(1,2).reshape(B,self.embed_dim,H,W)
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

