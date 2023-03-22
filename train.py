import torch
import torch.nn.functional as F
from tqdm import tqdm
from xxSOD import M3Net
from dataloader import get_loader
from data.dataloader import RGB_Dataset
import os
from torch.optim.lr_scheduler import _LRScheduler
from Models.layers import ImagePyramid
# IoU Loss
def iou_loss(pred, mask):
    pred  = torch.sigmoid(pred)
    inter = (pred*mask).sum(dim=(2,3))
    union = (pred+mask).sum(dim=(2,3))
    iou  = 1-(inter+1)/(union-inter+1)
    return iou.mean()

def structure_loss(pred, mask, weight=None):
    def generate_smoothed_gt(gts):
        epsilon = 0.001
        new_gts = (1-epsilon)*gts+epsilon/2
        return new_gts
    if weight == None:
        weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    else:
        weit = 1 + 5 * weight

    new_gts = generate_smoothed_gt(mask)
    wbce = F.binary_cross_entropy_with_logits(pred, new_gts, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()

def wbce(pred,mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    return wbce.mean()

def adaptive_pixel_intensity_loss(pred, mask):
    w1 = torch.abs(F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) - mask)
    w2 = torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
    w3 = torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)

    omega = 1 + 0.5 * (w1 + w2 + w3) * mask

    bce = F.binary_cross_entropy(pred, mask, reduce=None)
    abce = (omega * bce).sum(dim=(2, 3)) / (omega + 0.5).sum(dim=(2, 3))

    inter = ((pred * mask) * omega).sum(dim=(2, 3))
    union = ((pred + mask) * omega).sum(dim=(2, 3))
    aiou = 1 - (inter + 1) / (union - inter + 1)

    mae = F.l1_loss(pred, mask, reduce=None)
    amae = (omega * mae).sum(dim=(2, 3)) / (omega - 1).sum(dim=(2, 3))

    return (0.7 * abce + 0.7 * aiou + 0.7 * amae).mean()
def train_one_epoch(epoch,epochs,model,opt,scheduler,train_dl,train_size):
    epoch_total_loss = 0
    epoch_loss0 = 0
    epoch_loss1 = 0
    epoch_loss2 = 0
    epoch_loss3 = 0
    epoch_loss4 = 0
    imagepyramid = ImagePyramid(7,1)
    loss_weights = [1, 1, 1, 1, 1]
    l = 0

    progress_bar = tqdm(train_dl, desc='Epoch[{:03d}/{:03d}]'.format(epoch+1, epochs),ncols=140)
    for i, data_batch in enumerate(progress_bar):

        l = l+1

        images = data_batch['image']
        label = data_batch['gt']
        H,W = train_size
        images, label = images.cuda(non_blocking=True), label.cuda(non_blocking=True)

        #label = F.interpolate(label, (H//2,W//2), mode='nearest')
        #label = F.interpolate(label, (H//4,W//4), mode='nearest')
        #label = F.interpolate(label, (H//8,W//8), mode='nearest')

        mask_1_8, mask_1_4, mask_1_2, mask_1_1 = model(images)
        mask_1_8 = F.interpolate(mask_1_8,(H,W),mode='bilinear')
        mask_1_4 = F.interpolate(mask_1_4,(H,W),mode='bilinear')
        mask_1_2 = F.interpolate(mask_1_2,(H,W),mode='bilinear')
        #loss4  = F.binary_cross_entropy_with_logits(mask_1_16, label_1_16) + iou_loss(mask_1_16, label_1_16)
        loss3  = wbce(mask_1_8, label) + iou_loss(mask_1_8, label)
        loss2  = wbce(mask_1_4, label) + iou_loss(mask_1_4, label)
        loss1  = wbce(mask_1_2, label) + iou_loss(mask_1_2, label)
        loss0  = wbce(mask_1_1, label) + iou_loss(mask_1_1, label)

        loss = loss_weights[0] * loss0 + loss_weights[0] * loss1 + loss_weights[1] * loss2 + loss_weights[2] * loss3 #+ loss_weights[3] * loss4

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        epoch_total_loss += loss.cpu().data.item()
        epoch_loss0 += loss0.cpu().data.item()
        epoch_loss1 += loss1.cpu().data.item()
        epoch_loss2 += loss2.cpu().data.item()
        epoch_loss3 += loss3.cpu().data.item()
        #epoch_loss4 += loss4.cpu().data.item()

        progress_bar.set_postfix(loss=f'{epoch_loss0/(i+1):.3f}')
    return epoch_loss1/l
        
def fit(model, train_dl, epochs=60, lr=1e-4,train_size = 384):
    save_dir = './loss.txt'
    opt = get_opt(lr,model)
    scheduler = PolyLr(opt,gamma=0.9,minimum_lr=1.0e-07,max_iteration=len(train_dl)*epochs,warmup_iteration=12000)

    print('Starting train.')
    print('lr: '+str(lr))
    for epoch in range(epochs):
        #model.train()
        loss = train_one_epoch(epoch,epochs,model,opt,scheduler,train_dl,[train_size,train_size])
        fh = open(save_dir, 'a')
        fh.write(str(epoch+1) + ' epoch_loss: ' + str(loss) + '\n')
        fh.close()


def get_opt(lr,model):
    
    base_params = [params for name, params in model.named_parameters() if ("encoder" in name)]
    other_params = [params for name, params in model.named_parameters() if ("encoder" not in name)]
    params = [{'params': base_params, 'lr': lr*0.1},
          {'params': other_params, 'lr': lr}
         ]
         
    opt = torch.optim.Adam(params=params, lr=lr,weight_decay=0.0)

    return opt

class PolyLr(_LRScheduler):
    def __init__(self, optimizer, gamma, max_iteration, minimum_lr=0, warmup_iteration=0, last_epoch=-1):
        self.gamma = gamma
        self.max_iteration = max_iteration
        self.minimum_lr = minimum_lr
        self.warmup_iteration = warmup_iteration
        
        self.last_epoch = None
        self.base_lrs = []

        super(PolyLr, self).__init__(optimizer, last_epoch)

    def poly_lr(self, base_lr, step):
        return (base_lr - self.minimum_lr) * ((1 - (step / self.max_iteration)) ** self.gamma) + self.minimum_lr

    def warmup_lr(self, base_lr, alpha):
        return base_lr * (1 / 10.0 * (1 - alpha) + alpha)

    def get_lr(self):
        if self.last_epoch < self.warmup_iteration:
            alpha = self.last_epoch / self.warmup_iteration
            lrs = [min(self.warmup_lr(base_lr, alpha), self.poly_lr(base_lr, self.last_epoch)) for base_lr in
                    self.base_lrs]
        else:
            lrs = [self.poly_lr(base_lr, self.last_epoch) for base_lr in self.base_lrs]

        return lrs
    
def training(args):
    model = M3Net(embed_dim=512,dim=128,img_size=args.img_size,method=args.method)
    model.encoder.load_state_dict(torch.load('/mnt/ssd/yy/pretrained_model/swin_base_patch4_window12_384_22k.pth', map_location='cpu')['model'], strict=False)

    print('Pre-trained weight loaded.')

    #train_dataset = get_loader(args.trainset, args.data_root, 384, mode='train')
    train_dataset = RGB_Dataset(root=args.data_root, sets=['DUTS-TR'],img_size=args.img_size,mode='train')
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle = True, 
                                               pin_memory=True,num_workers = 4
                                               )
    
    model.cuda()
    model.train()
    print('Starting train.')
    fit(model,train_dl,args.train_epochs,args.lr,args.img_size)
    if not os.path.exists(args.save_model):
        os.makedirs(args.save_model)
    torch.save(model.state_dict(), args.save_model+args.method+'.pth')
    print('Saved as '+args.save_model+args.method+'.pth.')
