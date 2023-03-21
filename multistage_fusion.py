import torch
import torch.nn as nn
from timm.models.layers import DropPath
from Models.modules import MixedAttentionBlock,Conv2d
import torch.nn.functional as F
from Models.layers import *
from Models.context_module import *
from Models.attention_module import *
from Models.decoder_module import *

class decoder(nn.Module):
    r""" Multistage decoder. 
    
    Args:
        embed_dim (int): Dimension for attention. Default 384
        dim (int): Patch embedding dimension. Default 96
        img_size (int): Input image size. Default 224
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    """
    def __init__(self,in_channels = [128,128,256,512,1024],depth=64, base_size=[384, 384], threshold=512, **kwargs):
        super(decoder, self).__init__()
        #self.backbone = backbone
        self.in_channels = in_channels
        self.depth = depth
        self.base_size = base_size
        self.threshold = threshold
        
        self.context1 = PAA_e(self.in_channels[0], self.depth, base_size=self.base_size, stage=0)
        self.context2 = PAA_e(self.in_channels[1], self.depth, base_size=self.base_size, stage=1)
        self.context3 = PAA_e(self.in_channels[2], self.depth, base_size=self.base_size, stage=2)
        self.context4 = PAA_e(self.in_channels[3], self.depth, base_size=self.base_size, stage=3)
        self.context5 = PAA_e(self.in_channels[4], self.depth, base_size=self.base_size, stage=4)

        self.decoder = PAA_d(self.depth * 3, depth=self.depth, base_size=base_size, stage=2)

        self.attention0 = SICA(self.depth    , depth=self.depth, base_size=self.base_size, stage=0, lmap_in=True)
        self.attention1 = SICA(self.depth * 2, depth=self.depth, base_size=self.base_size, stage=1, lmap_in=True)
        self.attention2 = SICA(self.depth * 2, depth=self.depth, base_size=self.base_size, stage=2              )

        #self.pc_loss_fn  = nn.L1Loss()

        self.transition0 = Transition(17)
        self.transition1 = Transition(9)
        self.transition2 = Transition(5)
        
        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')
        self.image_pyramid = ImagePyramid(7, 1)

    def to(self, device):
        self.image_pyramid.to(device)
        #self.transition0.to(device)
        #self.transition1.to(device)
        #self.transition2.to(device)
        super(decoder, self).to(device)
        return self
    
    def cuda(self, idx=None):
        if idx is None:
            idx = torch.cuda.current_device()
            
        self.to(device="cuda:{}".format(idx))
        return self
    

    
    def forward(self, x):
        H, W = self.base_size
    
        x5,x4,x3,x2,x1 = x
        
        x1 = self.context1(x1) #4
        x2 = self.context2(x2) #4
        x3 = self.context3(x3) #8
        x4 = self.context4(x4) #16
        x5 = self.context5(x5) #32

        f3, d3 = self.decoder([x3, x4, x5]) #16

        f3 = self.res(f3, (H // 4,  W // 4 ))
        f2, p2 = self.attention2(torch.cat([x2, f3], dim=1), d3)
        d2 = self.image_pyramid.reconstruct(d3, p2) #4

        x1 = self.res(x1, (H // 2, W // 2))
        f2 = self.res(f2, (H // 2, W // 2))
        f1, p1 = self.attention1(torch.cat([x1, f2], dim=1), d2, p2) #2
        d1 = self.image_pyramid.reconstruct(d2, p1) #2
        
        f1 = self.res(f1, (H, W))
        _, p0 = self.attention0(f1, d1, p1) #2
        d0 = self.image_pyramid.reconstruct(d1, p0) #2
        '''
        xx = p0.detach().cpu().squeeze()
        xx = xx-xx.min()
        xx = xx/xx.max()*255
        cv2.imwrite('1.png',np.asarray(xx))
        xx = d0.detach().cpu().squeeze()
        xx = xx-xx.min()
        xx = xx/xx.max()*255
        cv2.imwrite('2.png',np.asarray(xx))
        '''        
        out = [d3,d2,d1,d0]
        

        return out
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


