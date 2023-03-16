import torch
import torch.nn as nn
from timm.models.layers import DropPath
from Models.modules import CrossAttention
class MultilevelInteractionBlock(nn.Module):
    r""" Multilevel Interaction Block. 
    
    Args:
        dim (int): Number of low-level feature channels.
        dim1, dim2 (int): Number of high-level feature channels.
        embed_dim (int): Dimension for attention.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    """
    
    def __init__(self,dim,dim1,dim2=None,embed_dim = 384,num_heads = 6,mlp_ratio = 3., qkv_bias = False, qk_scale = None,drop = 0.,attn_drop = 0.,drop_path = 0.,act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(MultilevelInteractionBlock, self).__init__()
        self.interact1 = CrossAttention(dim1 = dim,dim2 = dim1,dim = embed_dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,attn_drop=attn_drop,proj_drop=drop)
        self.norm0 = norm_layer(dim)
        self.norm1 = norm_layer(dim1)
        self.dim = dim
        self.dim2 = dim2
        self.mlp_ratio = mlp_ratio
        if self.dim2:
            self.interact2 = CrossAttention(dim1 = dim,dim2 = dim2,dim = embed_dim,num_heads=num_heads,qkv_bias=qkv_bias,qk_scale=qk_scale,attn_drop=attn_drop,proj_drop=drop)
            self.norm2 = norm_layer(dim2)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*mlp_ratio),
            act_layer(),
            nn.Linear(dim*mlp_ratio, dim),
        )

    def forward(self,fea,fea_1,fea_2=None):
        fea = self.norm0(fea)
        fea_1 = self.norm1(fea_1)
        fea_1 = self.interact1(fea,fea_1)
        if self.dim2:
            fea_2 = self.norm2(fea_2)
            fea_2 = self.interact2(fea,fea_2)
        fea = fea + fea_1    
        if self.dim2:
            fea = fea + fea_2    
        fea = fea + self.drop_path(self.mlp(self.norm(fea)))
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
    model = MultilevelInteractionBlock(dim1=96,dim2=192,dim3=384)
    model.cuda()
    f = []
    f.append(torch.randn((1,3136,96)).cuda())
    f.append(torch.randn((1,784,192)).cuda())
    f.append(torch.randn((1,196,384)).cuda())
    y = model(f[0],f[1],f[2])
    print(y.shape)

