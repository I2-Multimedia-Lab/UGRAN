import torch
import torch.nn as nn
from lib.swin import SwinTransformer
from lib.resnet import ResNet
from lib.t2t_vit import T2t_vit_t_14
from lib.res2net_v1b import res2net50_v1b_26w_4s
from multistage_fusion import decoder
from lib.modules import weight_init
import time

class UGRAN(nn.Module):
    r""" Uncertainty Guided Refinement for Fine-grained Salient Object Detection. 
    
    Args:
        embed_dim (int): Dimension for attention. Default 384
        dim (int): Patch embedding dimension. Default 96
        img_size (int): Input image size. Default 224
        method (string): Backbone used as the encoder.
    """

    def __init__(self, dim = 64, img_size = 384, method = 'UGRAN-S', mode = 'train'):
        super(UGRAN, self).__init__()
        self.window_size = img_size // 32
        self.img_size = img_size
        self.feature_dims = []
        self.method = method
        self.dim = dim
        self.mode = mode
        if method == 'UGRAN-S':
            self.encoder = SwinTransformer(pretrain_img_size = img_size, 
                                        embed_dim = 128,
                                        depths = [2,2,18,2],
                                        num_heads = [4,8,16,32],
                                        window_size = self.window_size)
            self.decoder = decoder(in_channels = [128,128,256,512,1024], dim = dim, base_size = [img_size,img_size], window_size = self.window_size)

        elif method == 'UGRAN-R':
            self.encoder = ResNet()
            self.decoder = decoder(in_channels = [64,256,512,1024,2048], dim = dim, base_size = [img_size,img_size], window_size = self.window_size)

        elif method == 'UGRAN-R2':
            self.encoder = res2net50_v1b_26w_4s()
            self.decoder = decoder(in_channels = [64,256,512,1024,2048], dim = dim, base_size = [img_size,img_size], window_size = self.window_size)

        self.initialize()
        self.ptime = 0.0 # partition time
        self.rtime = 0.0 # reverse time
        self.etime = 0.0 # execute time

    def forward(self,x):
        st = time.process_time()

        fea = self.encoder(x)
        fea_0,fea_1_4,fea_1_8,fea_1_16,fea_1_32 = fea

        mask = self.decoder([fea_0,fea_1_4,fea_1_8,fea_1_16,fea_1_32],self.mode)
        et = time.process_time()
        self.etime=(et-st)
        return mask#, self.etime
    
    def to(self, device):

        self.encoder.to(device)
        self.decoder.to(device)
        super(UGRAN, self).to(device)
        return self
    
    def cuda(self, idx=None):
        if idx is None:
            idx = torch.cuda.current_device()
            
        self.to(device="cuda:{}".format(idx))
        return self
        
    def initialize(self):
        weight_init(self)
    def flops(self):
        flops = 0
        flops += self.encoder.flops()
        N1 = self.img_size//4*self.img_size//4
        N2 = self.img_size//8*self.img_size//8
        N3 = self.img_size//16*self.img_size//16
        N4 = self.img_size//32*self.img_size//32
        flops += self.interact1.flops(N3,N4)
        flops += self.interact2.flops(N2,N3,N4)
        flops += self.interact3.flops(N1,N2,N3)
        flops += self.decoder.flops()
        return flops

#from thop import profile
if __name__ == '__main__':
    # Test
    model = UGRAN(dim=64,img_size=384,method='UGRAN-R',mode='train')
    #model.encoder.load_state_dict(torch.load('/mnt/ssd/yy/pretrained_model/resnet50.pth'), strict=False)
                                             #swin_base_patch4_window12_384_22k.pth', map_location='cpu')['model'], strict=False)
    #model.load_state_dict(torch.load('/mnt/ssd/yy/xxSOD/savepth/UGRAN-R_single.pth'))
    model.cuda()
    
    f = torch.randn((1,3,384,384))
    x = model(f.cuda())
    print(len(x))
    for m in x:
        print(m.shape)
    '''print(model.decoder.attention0.ptime)
    print(model.decoder.attention0.rtime)
    print(model.decoder.attention0.etime)
    print(model.decoder.attention1.ptime)
    print(model.decoder.attention1.rtime)
    print(model.decoder.attention1.etime)
    print(model.decoder.attention2.ptime)
    print(model.decoder.attention2.rtime)
    print(model.decoder.attention2.etime)'''
    import torch
    from ptflops import get_model_complexity_info

    #macs, params = get_model_complexity_info(model, (3, 384, 384), as_strings=True, print_per_layer_stat=True, verbose=True)

    #print('{:<30}  {:<8}'.format('macs: ', macs))
    #print('{:<30}  {:<8}'.format('parameters: ', params))
    