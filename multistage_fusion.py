import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.scale_spatial_consistent_attention import SSCA
from lib.uncertainty_aware_refine_attention import URA
from lib.multilevel_interaction_attention import MIA
from lib.modules import *
import time

class decoder(nn.Module):
    r""" Multistage decoder. 
    
    Args:
        in_channels (tuple): Number of input feature channels. 
        dim (int): Patch embedding dimension. Default 64
        base_size (int): Input image size. Default 384

    """
    def __init__(self, in_channels = [128, 128, 256, 512, 1024], dim = 64, base_size = [384, 384]):
        super(decoder, self).__init__()
        self.in_channels = in_channels
        self.dim = dim
        self.base_size = base_size

        self.context5 = MIA(in_channel = in_channels[4], out_channel = dim,dim1 = None,dim2 = None, embed_dim = dim * 16, num_heads = 8, mlp_ratio = 3)
        self.context4 = MIA(in_channel = in_channels[3], out_channel = dim, dim1 = in_channels[4], dim2 = None, embed_dim =dim * 8, num_heads = 4, mlp_ratio = 3)
        self.context3 = MIA(in_channel = in_channels[2], out_channel = dim, dim1 = in_channels[3], dim2 = in_channels[4], embed_dim=dim * 4, num_heads = 2, mlp_ratio= 3)
        self.context2 = MIA(in_channel = in_channels[1], out_channel = dim, dim1 = in_channels[2], dim2 = in_channels[3], dim3 = in_channels[4], embed_dim = dim * 2,num_heads = 1, mlp_ratio = 3)
        if(self.dim != self.in_channels[0]):
            self.context1 = Conv2d(in_channels[0],dim,1)

        self.fusion4 = SSCA(in_channel = dim * 2, dim = dim, embed_dim = self.dim * 8, num_heads = 4, stacked = 1, stage = 4)
        self.fusion3 = SSCA(in_channel = dim * 2, dim = dim, embed_dim = self.dim * 4, num_heads = 2, stacked = 1, stage = 3)
        self.fusion2 = SSCA(in_channel = dim * 2, dim = dim, embed_dim = self.dim * 2, num_heads = 1, stacked = 1, stage = 2)

        self.attention0 = URA(self.dim, dim = self.dim, base_size = self.base_size, stage = 0)
        self.attention1 = URA(self.dim, dim = self.dim, base_size = self.base_size, stage = 1)
        self.attention2 = URA(self.dim, dim = self.dim, base_size = self.base_size, stage = 2)

        self.image_pyramid = ImagePyramid(7, 1)
        self.uthreshold = 0.5
        self.ptime = 0.0 # partition time
        self.utime = 0.0 # reverse time
        self.etime = 0.0 # execute time

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

    def forward(self, x, mode):

        self.ptime = 0.0 # partition time
        self.utime = 0.0 # reverse time
        self.etime = 0.0
        H, W = self.base_size
        x1_, x2_, x3_, x4_, x5_ = x
        x5 = self.context5(x5_) #32
        x4 = self.context4(x4_, fea_1 = x5_)#,x_h=x5) #16
        x3 = self.context3(x3_, fea_1 = x4_)#,x_h=x4) #8
        x2 = self.context2(x2_, fea_1 = x3_)#,x_h=x3) #4
        if(self.dim == self.in_channels[0]):
            l = x1_
        else:
            l = self.context1(x1_) #4

        f5 = F.interpolate(x5,(H // 16, W // 16), mode = 'bilinear', align_corners = False)
        f4, s4 = self.fusion4(torch.cat([x4, f5], dim = 1))

        f4 = F.interpolate(f4,(H // 8, W // 8), mode = 'bilinear', align_corners = False)
        f3, s3 = self.fusion3(torch.cat([x3, f4], dim = 1))

        f3 = F.interpolate(f3, (H // 4, W // 4), mode = 'bilinear', align_corners = False)
        f2, s2 = self.fusion2(torch.cat([x2, f3], dim = 1))
        c2 = self.image_pyramid.get_uncertain(s2, (H // 4, W // 4))

        st = time.process_time()
        f2, r2, p2, p, e = self.attention2(f2, l.detach(), c2.detach())
        et = time.process_time()
        self.utime += (et - st)
        self.ptime += p
        self.etime += e
        d2 = self.image_pyramid.reconstruct(s2.detach(), r2) 

        #f2 = F.interpolate(f2, (H//2, W //2), mode='bilinear', align_corners=False)
        #l = F.interpolate(l, (H//2, W//2), mode='bilinear', align_corners=False)
        c2 = self.image_pyramid.get_uncertain(d2, (H // 4, W // 4))
        st = time.process_time()
        f1, r1, p1, p, e = self.attention1(f2, l.detach(), c2.detach()) 
        et = time.process_time()
        self.utime += (et - st)
        self.ptime += p
        self.etime += e
        d1 = self.image_pyramid.reconstruct(d2.detach(), r1) 

        #f1 = F.interpolate(f1, (H, W), mode='bilinear', align_corners=False)
        #l = F.interpolate(l, (H, W), mode='bilinear', align_corners=False)
        c1 = self.image_pyramid.get_uncertain(d1, (H // 4, W // 4))
        st = time.process_time()
        _, r0, p0, p, e = self.attention0(f1,l.detach(), c1.detach()) #2
        et = time.process_time()
        self.utime += (et - st)
        self.ptime += p
        self.etime += e
        d0 = self.image_pyramid.reconstruct(d1.detach(), r0) 

        c0 = self.image_pyramid.get_uncertain(d2, (H, W))
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
        
        out = p2, p1 ,p0 ,c2 ,c1 ,c0 ,s4 ,s3 ,s2 ,r2 ,r1 ,r0,d2, d1, d0
    
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


