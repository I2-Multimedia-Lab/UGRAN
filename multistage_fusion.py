import torch
import torch.nn as nn
from timm.models.layers import DropPath
from lib.modules import MixedAttentionBlock
import torch.nn.functional as F
#from lib.layers import *
from lib.multiscale_feature_enhancement import MFE
from lib.scale_spatial_consistent_attention import SSCA
from lib.uncertainty_aware_refine_attention import URA
from lib.multilevel_interaction_attention import MIA
from lib.modules import *
class decoder(nn.Module):
    r""" Multistage decoder. 
    
    Args:
        embed_dim (int): Dimension for attention. Default 384
        dim (int): Patch embedding dimension. Default 96
        img_size (int): Input image size. Default 224
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    """
    def __init__(self,in_channels = [128,128,256,512,1024],depth=64, base_size=[384, 384], window_size = 12, mode='train'):
        super(decoder, self).__init__()
        self.in_channels = in_channels
        self.depth = depth
        self.base_size = base_size
        self.mode = mode

        self.context5 = MIA(in_channel=in_channels[4],out_channel=depth,dim1=None,dim2=None,embed_dim=depth*16,num_heads=8,mlp_ratio=3)
        self.context4 = MIA(in_channel=in_channels[3],out_channel=depth,dim1=in_channels[4],dim2=None,embed_dim=depth*8,num_heads=4,mlp_ratio=3)
        self.context3 = MIA(in_channel=in_channels[2],out_channel=depth,dim1=in_channels[3],dim2=in_channels[4],embed_dim=depth*4,num_heads=2,mlp_ratio=3)
        self.context2 = MIA(in_channel=in_channels[1],out_channel=depth,dim1=in_channels[2],dim2=in_channels[3],dim3=in_channels[4],embed_dim=depth*2,num_heads=1,mlp_ratio=3)

        self.fusion4 = SSCA(in_channel=depth*2,depth=depth,dim=self.depth*8,num_heads=4,stacked=1,stage=4)
        self.fusion3 = SSCA(in_channel=depth*2,depth=depth,dim=self.depth*4,num_heads=2,stacked=1,stage=3)
        self.fusion2 = SSCA(in_channel=depth*2,depth=depth,dim=self.depth*2,num_heads=1,stacked=1,stage=2)
        self.proj = Conv2d(depth,1,1)

        self.attention0 = URA(self.depth, depth=self.depth, base_size=self.base_size, window_size=window_size,c_num=3, stage=0)
        self.attention1 = URA(self.depth, depth=self.depth, base_size=self.base_size, window_size=window_size,c_num=3, stage=1)
        self.attention2 = URA(self.depth, depth=self.depth, base_size=self.base_size, window_size=window_size,c_num=3, stage=2)

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')

        self.image_pyramid = ImagePyramid(15, 1)
        self.uthreshold = 0.5

        #self.initialize()

    def to(self, device):
        self.image_pyramid.to(device)
        super(decoder, self).to(device)
        return self
    
    def cuda(self, idx=None):
        if idx is None:
            idx = torch.cuda.current_device()
            
        self.to(device="cuda:{}".format(idx))
        return self
    
    def initialize(self):
        weight_init(self)

    def forward(self, x):
        H, W = self.base_size
    
        x1_,x2_,x3_,x4_,x5_ = x
        l = x1_
        
        x5 = self.context5(x5_) #32
        x4 = self.context4(x4_,fea_1=x5_)#,x_h=x5) #16
        x3 = self.context3(x3_,fea_1=x4_)#,x_h=x4) #8
        x2 = self.context2(x2_,fea_1=x3_)#,x_h=x3) #4
        #x1 = self.context1(x1,x_h=x2) #4

        '''
        f3, d3 = self.decoder([x3, x4, x5]) #16
        '''
        f5 = self.res(x5,(H//16,W//16))
        f4, s4 = self.fusion4(torch.cat([x4,f5],dim=1))

        f4 = self.res(f4,(H//8,W//8))
        f3, s3 = self.fusion3(torch.cat([x3,f4],dim=1))

        f3 = self.res(f3, (H//4, W//4 ))
        f2, s2 = self.fusion2(torch.cat([x2,f3],dim=1))
        c2 = self.image_pyramid.get_uncertain(s2,(H//4, W//4))
        f2, r2, p2 = self.attention2(f2, l.detach(), c2.detach())
        d2 = self.image_pyramid.reconstruct(s2.detach(), r2) 
        if self.mode == 'train':
            #f2 = self.res(f2, (H // 2, W // 2))
            #l = self.res(l, (H // 2, W // 2))
            c2 = self.image_pyramid.get_uncertain(d2,(H//4, W //4))
            f1, r1, p1 = self.attention1(f2,l.detach(),c2.detach()) 
            d1 = self.image_pyramid.reconstruct(d2.detach(), r1) 
            #f1 = self.res(f1, (H, W))
            #l = self.res(l, (H, W))
            c1 = self.image_pyramid.get_uncertain(d1,(H//4, W//4))
            _, r0, p0 = self.attention0(f1,l.detach(),c1.detach()) #2
            d0 = self.image_pyramid.reconstruct(d1.detach(), r0) 
        else:
            f2 = self.res(f2, (H//2, W //2))
            l = self.res(l, (H//2, W//2))
            c2 = self.image_pyramid.get_uncertain(d2,(H//2, W//2))
            f1, r1, p1 = self.attention1(f2,l.detach(),c2.detach()) 
            d1 = self.image_pyramid.reconstruct(d2.detach(), r1) 
            f1 = self.res(f1, (H, W))
            l = self.res(l, (H, W))
            c1 = self.image_pyramid.get_uncertain(d1,(H, W))
            _, r0, p0 = self.attention0(f1,l.detach(),c1.detach()) #2
            d0 = self.image_pyramid.reconstruct(d1.detach(), r0) 

        c0 = self.image_pyramid.get_uncertain(d2,(H, W))
        '''
        xx = p1.detach().cpu().squeeze()
        xx = xx-xx.min()
        xx = xx/xx.max()*255
        cv2.imwrite('1.png',np.asarray(xx))
        xx = d1.detach().cpu().squeeze()
        xx = xx-xx.min()
        xx = xx/xx.max()*255
        cv2.imwrite('2.png',np.asarray(xx))
        ''' 
        
        out = [p2,p1,p0,c2,c1,c0,s4,s3,s2,r2,r1,r0,d2,d1,d0]
    
        return out


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


