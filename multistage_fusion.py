import torch
import torch.nn as nn
from timm.models.layers import DropPath
from lib.modules import MixedAttentionBlock
import torch.nn.functional as F
#from lib.layers import *
from lib.context_module import *
from lib.attention_module import *
from lib.decoder_module import *
from lib.multiscale_feature_enhancement import MFE
from lib.scale_spatial_consistent_attention import SSCA
from lib.window_based_context_attention import WCA
from lib.modules import *
class decoder(nn.Module):
    r""" Multistage decoder. 
    
    Args:
        embed_dim (int): Dimension for attention. Default 384
        dim (int): Patch embedding dimension. Default 96
        img_size (int): Input image size. Default 224
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
    """
    def __init__(self,in_channels = [128,128,256,512,1024],depth=64, base_size=[384, 384], window_size = 12):
        super(decoder, self).__init__()
        self.in_channels = in_channels
        self.depth = depth
        self.base_size = base_size
        
        #self.context1 = MFE(in_channel=self.in_channels[0],h_channel=self.in_channels[1], out_channel = self.depth, base_size=self.base_size, stage=2)
        self.context2 = MFE(in_channel=self.in_channels[1],l_channel=self.in_channels[0],h_channel=self.in_channels[2], out_channel=self.depth, base_size=self.base_size, stage=2)
        self.context3 = MFE(in_channel=self.in_channels[2],l_channel=self.in_channels[1],h_channel=self.in_channels[3], out_channel=self.depth, base_size=self.base_size, stage=3)
        self.context4 = MFE(in_channel=self.in_channels[3],l_channel=self.in_channels[2],h_channel=self.in_channels[4], out_channel=self.depth, base_size=self.base_size, stage=4)
        self.context5 = MFE(in_channel=self.in_channels[4],l_channel=self.in_channels[3],out_channel=self.depth, base_size=self.base_size, stage=5)

        #self.decoder = PAA_d(self.depth * 3, depth=self.depth, base_size=base_size, stage=2)
        self.fusion4 = SSCA(self.depth*2,dim=self.in_channels[3],num_heads=4,depth=self.depth,stage=4)
        self.fusion3 = SSCA(self.depth*2,dim=self.in_channels[2],num_heads=2,depth=self.depth,stage=3)
        self.fusion2 = SSCA(self.depth*2,dim=self.in_channels[1],num_heads=1,depth=self.depth,stage=2)
        #self.fusion1 = SSCA(self.depth*2,dim=self.in_channels[1],depth=self.depth,stage=1)
        self.proj = Conv2d(depth,1,1)

        self.attention0 = WCA(self.depth, depth=self.depth, base_size=self.base_size, window_size=window_size,c_num=3, stage=0)
        self.attention1 = WCA(self.depth, depth=self.depth, base_size=self.base_size, window_size=window_size,c_num=3, stage=1)
        self.attention2 = WCA(self.depth, depth=self.depth, base_size=self.base_size, window_size=window_size,c_num=3, stage=2)

        #self.pc_loss_fn  = nn.L1Loss()

        self.ret = lambda x, target: F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
        self.res = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        self.des = lambda x, size: F.interpolate(x, size=size, mode='nearest')

        #self.initialize()

    '''
    def to(self, device):
        #self.image_pyramid.to(device)
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
    '''
    def initialize(self):
        weight_init(self)
    
    def forward(self, x):
        H, W = self.base_size
    
        x1,x2,x3,x4,x5 = x
        
        #x1 = self.context1(x1,x_h=x2) #4
        x2 = self.context2(x2)#,x_h=x3) #4
        x3 = self.context3(x3)#,x_h=x4) #8
        x4 = self.context4(x4)#,x_h=x5) #16
        x5 = self.context5(x5) #32

        '''
        f3, d3 = self.decoder([x3, x4, x5]) #16
        '''
        f5 = self.res(x5,(H//16,W//16))
        f4, d4 = self.fusion4(torch.cat([x4,f5],dim=1))

        f4 = self.res(f4,(H//8,W//8))
        f3, d3 = self.fusion3(torch.cat([x3,f4],dim=1))

        f3 = self.res(f3, (H // 4,  W // 4 ))
        f2, d2 = self.fusion2(torch.cat([x2,f3],dim=1))
        f2, d2 = self.attention2(f2, d3)
        #d2 = self.res(d3, (H//4,W//4))+p2

        f2 = self.res(f2, (H // 2, W // 2))
        f1, d1 = self.attention1(f2,d2) #2
        #d1 = self.res(d2, (H//2,W//2))+p1

        f1 = self.res(f1, (H, W))
        _, d0 = self.attention0(f1,d1) #2
        #d0 = self.res(d1, (H,W))+p0

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
        
        out = [d3,d2,d1,d0]
    
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


