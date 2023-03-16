import torch
import torch.nn as nn
from timm.models.layers import DropPath
from Models.modules import MixedAttentionBlock
class decoder(nn.Module):
    r""" Multistage decoder. 
    
    Args:
        embed_dim (int): Dimension for attention. Default 384
        dim (int): Patch embedding dimension. Default 96
        img_size (int): Input image size. Default 224
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    """
    def __init__(self,embed_dim=384,dims=[96,192,384],img_size=224,mlp_ratio=3):
        super(decoder, self).__init__()
        self.img_size = img_size
        #self.dim = dim
        self.embed_dim = embed_dim
        self.fusion1 = multiscale_fusion(in_dim=dims[2],f_dim=dims[1],kernel_size=(3,3),img_size=(img_size//8,img_size//8),stride=(2,2),padding=(1,1))
        self.fusion2 = multiscale_fusion(in_dim=dims[1],f_dim=dims[0],kernel_size=(3,3),img_size=(img_size//4,img_size//4),stride=(2,2),padding=(1,1))
        self.fusion3 = multiscale_fusion(in_dim=dims[0],f_dim=dims[0],kernel_size=(7,7),img_size=(img_size//1,img_size//1),stride=(4,4),padding=(2,2),fuse=False)

        self.mixatt1 = MixedAttention(in_dim=dims[1],dim=embed_dim,img_size=(img_size//8,img_size//8),num_heads=1,mlp_ratio=mlp_ratio,depth=2)
        self.mixatt2 = MixedAttention(in_dim=dims[0],dim=embed_dim,img_size=(img_size//4,img_size//4),num_heads=1,mlp_ratio=mlp_ratio,depth=2)

        self.proj1 = nn.Linear(dims[2],1)
        self.proj2 = nn.Linear(dims[1],1)
        self.proj3 = nn.Linear(dims[0],1)
        self.proj4 = nn.Linear(dims[0],1)



    def forward(self,f):
        fea_1_16,fea_1_8,fea_1_4 = f #fea_1_16:1/16
        B,_,_ = fea_1_16.shape
        fea_1_8 = self.fusion1(fea_1_16,fea_1_8)
        fea_1_8 = self.mixatt1(fea_1_8)
        fea_1_4 = self.fusion2(fea_1_8,fea_1_4)
        fea_1_4 = self.mixatt2(fea_1_4)
        fea_1_1 = self.fusion3(fea_1_4)
        fea_1_16 = self.proj1(fea_1_16)
        mask_1_16 = fea_1_16.transpose(1, 2).reshape(B, 1, self.img_size // 16, self.img_size // 16)
        fea_1_8 = self.proj2(fea_1_8)
        mask_1_8 = fea_1_8.transpose(1, 2).reshape(B, 1, self.img_size // 8, self.img_size // 8)
        fea_1_4 = self.proj3(fea_1_4)
        mask_1_4 = fea_1_4.transpose(1, 2).reshape(B, 1, self.img_size // 4, self.img_size // 4)
        fea_1_1 = self.proj4(fea_1_1)
        mask_1_1 = fea_1_1.transpose(1, 2).reshape(B, 1, self.img_size // 1, self.img_size // 1)
        return [mask_1_16,mask_1_8,mask_1_4,mask_1_1]
    
    def flops(self):
        flops = 0
        flops += self.fusion1.flops()
        flops += self.fusion2.flops()
        flops += self.fusion3.flops()
        flops += self.mixatt1.flops()
        flops += self.mixatt2.flops()
        
        flops += self.img_size//16*self.img_size//16 * self.dim * 4
        flops += self.img_size//8*self.img_size//8 * self.dim * 2
        flops += self.img_size//4*self.img_size//4 * self.dim * 1
        flops += self.img_size//1*self.img_size//1 * self.dim * 1

        return flops


class multiscale_fusion(nn.Module):
    r""" Upsampling and feature fusion. 
    
    Args:
        in_dim (int): Number of input feature channels.
        f_dim (int): Number of fusion feature channels.
        img_size (int): Image size after upsampling.
        kernel_size (tuple(int)): The size of the sliding blocks.
        stride (int): The stride of the sliding blocks in the input spatial dimensions, can be regarded as upsampling ratio. 
        padding (int): Implicit zero padding to be added on both sides of input. 
        fuse (bool): If True, concat features from different levels. 
    """
    def __init__(self,in_dim,f_dim,kernel_size,img_size,stride,padding,fuse=True):
        super(multiscale_fusion, self).__init__()
        self.fuse = fuse
        self.norm = nn.LayerNorm(in_dim)
        self.in_dim = in_dim
        self.f_dim = f_dim
        self.kernel_size = kernel_size
        self.img_size = img_size
        self.project = nn.Linear(in_dim, in_dim * kernel_size[0] * kernel_size[1])
        self.upsample = nn.Fold(output_size=img_size, kernel_size=kernel_size, stride=stride, padding=padding)
        if self.fuse:
            self.mlp1 = nn.Sequential(
                nn.Linear(in_dim+f_dim, f_dim),
                nn.GELU(),
                nn.Linear(f_dim, f_dim),
            )
        else:
            self.proj = nn.Linear(in_dim,f_dim)
        
    def forward(self,fea,fea_1=None):
        fea = self.project(self.norm(fea))
        fea = self.upsample(fea.transpose(1,2))
        B, C, _, _ = fea.shape
        fea = fea.view(B, C, -1).transpose(1, 2)#.contiguous()
        if self.fuse:
            fea = torch.cat([fea,fea_1],dim=2)
            fea = self.mlp1(fea)
        else:
            fea = self.proj(fea)
        return fea
    def flops(self):
        N = self.img_size[0]*self.img_size[1]
        flops = 0
        #norm
        flops += N * self.in_dim
        #proj
        flops += N*self.in_dim*self.in_dim*self.kernel_size[0]*self.kernel_size[1]
        #mlp
        flops += N*(self.in_dim+self.f_dim)*self.f_dim
        flops += N*self.f_dim*self.f_dim
        return flops
    
class MixedAttention(nn.Module):
    r""" Mixed Attention Module. 
    
    Args:
        in_dim (int): Number of input feature channels.
        dim (int): Number for attention. 
        img_size (int): Image size after upsampling.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        depth (int): The number of MAB stacked.
    """
    def __init__(self,in_dim,dim,img_size,num_heads=1,mlp_ratio=4,depth=2,drop_path = 0.):
        super(MixedAttention, self).__init__()

        self.img_size = img_size
        self.in_dim = in_dim
        self.dim = dim
        self.norm1 = nn.LayerNorm(in_dim)
        self.mlp1 = nn.Sequential(
            nn.Linear(in_dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.blocks = nn.ModuleList([
            MixedAttentionBlock(dim=dim,img_size=img_size,num_heads=num_heads,mlp_ratio=mlp_ratio)
            for i in range(depth)])
        self.norm2 = nn.LayerNorm(dim)
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, in_dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self,fea):
        fea = self.mlp1(self.norm1(fea))
        for blk in self.blocks:
            fea = blk(fea)
        fea = self.drop_path(self.mlp2(self.norm2(fea)))
        return fea
    def flops(self):
        flops = 0
        N = self.img_size[0]*self.img_size[1]
        #norm1
        flops += N*self.in_dim
        #mlp1
        flops += N*self.in_dim*self.dim
        flops += N*self.dim*self.dim

        #blks
        for blk in self.blocks:
            flops += blk.flops()
        #norm2
        flops += N*self.dim
        #mlp2
        flops += N*self.in_dim*self.dim
        flops += N*self.dim*self.dim
        return flops


if __name__ == '__main__':
    # Test
    model = decoder(embed_dim=384,dim=96,img_size=224)
    model.cuda()
    f = []
    f.append(torch.randn((1,196,384)).cuda())
    f.append(torch.randn((1,784,192)).cuda())
    f.append(torch.randn((1,3136,96)).cuda())

    y = model(f)
    print(y[0].shape)
    print(y[1].shape)
    print(y[2].shape)
    print(y[3].shape)


