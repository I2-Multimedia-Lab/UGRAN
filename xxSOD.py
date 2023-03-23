import torch
import torch.nn as nn
from Models.swin import SwinTransformer
from Models.resnet import ResNet
from Models.t2t_vit import T2t_vit_t_14
from multistage_fusion import decoder
from multilevel_interaction import MultilevelInteractionBlock
class M3Net(nn.Module):
    r""" Multilevel, Mixed and Multistage Attention Network for Salient Object Detection. 
    
    Args:
        embed_dim (int): Dimension for attention. Default 384
        dim (int): Patch embedding dimension. Default 96
        img_size (int): Input image size. Default 224
        method (string): Backbone used as the encoder.
    """

    def __init__(self,embed_dim=384,dim=96,img_size=384,method='M3Net-S'):
        super(M3Net, self).__init__()
        if img_size == 384:
            self.window_size=12
        else:
            self.window_size=7
        self.img_size = img_size
        self.feature_dims = []
        self.method = method
        self.dim = dim
        self.encoder = SwinTransformer(pretrain_img_size=img_size, 
                                        embed_dim=dim,
                                        depths=[2,2,18,2],
                                        num_heads=[4,8,16,32],
                                        window_size=self.window_size)

        #feature_dims=[dim,dim*2,dim*4,dim*8]
        self.decoder = decoder(base_size=[img_size,img_size],window_size=self.window_size)

    def forward(self,x):
        fea = self.encoder(x)
        fea_0,fea_1_4,fea_1_8,fea_1_16,fea_1_32 = fea

        mask = self.decoder([fea_0,fea_1_4,fea_1_8,fea_1_16,fea_1_32])
        return mask
    def to(self, device):
        #self.image_pyramid.to(device)
        #self.transition0.to(device)
        #self.transition1.to(device)
        #self.transition2.to(device)
        self.encoder.to(device)
        self.decoder.to(device)
        super(M3Net, self).to(device)
        return self
    
    def cuda(self, idx=None):
        if idx is None:
            idx = torch.cuda.current_device()
            
        self.to(device="cuda:{}".format(idx))
        return self
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
    model = M3Net(embed_dim=384,dim=128,img_size=384,method='M3Net-S')
    model.encoder.load_state_dict(torch.load('/mnt/ssd/yy/pretrained_model/swin_base_patch4_window12_384_22k.pth', map_location='cpu')['model'], strict=False)

    model.cuda()
    
    f = torch.randn((1,3,384,384))
    x = model(f.cuda())
    for m in x:
        print(m.shape)
    
    import torch
    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(model, (3, 384, 384), as_strings=True, print_per_layer_stat=True, verbose=True)

    print('{:<30}  {:<8}'.format('macs: ', macs))
    print('{:<30}  {:<8}'.format('parameters: ', params))
    