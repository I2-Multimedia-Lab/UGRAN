import torch
import torch.nn.functional as F
from tqdm import tqdm
from xxSOD import M3Net
from data.dataloader import RGB_Dataset
import os
from torch.optim.lr_scheduler import _LRScheduler
import datetime
from loss import *

def train_one_epoch(epoch,epochs,model,opt,scheduler,train_dl,train_size):
    epoch_total_loss = 0
    epoch_loss0 = 0
    epoch_loss1 = 0
    epoch_loss2 = 0
    epoch_loss3 = 0
    #epoch_loss4 = 0
    consistent_loss = nn.L1Loss()
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

        par_1_4,par_1_2,par_1_1,unc_1_4,unc_1_2,unc_1_1,sal_1_16, sal_1_8, sal_1_4, mask_1_4, mask_1_2, mask_1_1 = model(images)
        
        sal_1_16 = F.interpolate(sal_1_16,(H,W),mode='bilinear')
        sal_1_8 = F.interpolate(sal_1_8,(H,W),mode='bilinear')
        sal_1_4 = F.interpolate(sal_1_4,(H,W),mode='bilinear')

        mask_1_4 = F.interpolate(mask_1_4,(H,W),mode='bilinear')
        mask_1_2 = F.interpolate(mask_1_2,(H,W),mode='bilinear')
        mask_1_1 = F.interpolate(mask_1_1,(H,W),mode='bilinear')
        
        unc_1_4 = F.interpolate(unc_1_4,(H,W),mode='bilinear')
        unc_1_2 = F.interpolate(unc_1_2,(H,W),mode='bilinear')
        unc_1_1 = F.interpolate(unc_1_1,(H,W),mode='bilinear')

        loss4 = F.binary_cross_entropy_with_logits(sal_1_16, label) + iou_loss(sal_1_16, label)
        loss3 = F.binary_cross_entropy_with_logits(sal_1_8, label) + iou_loss(sal_1_8, label)
        loss2_ = F.binary_cross_entropy_with_logits(sal_1_4, label) + iou_loss(sal_1_4, label)
        #loss2 = structure_loss(mask_1_4, label, unc_1_4)#wbce(mask_1_4, label, unc_1_4) + iou_loss(mask_1_4, label)
        #loss1 = structure_loss(mask_1_2, label, unc_1_2)#wbce(mask_1_2, label, unc_1_2) + iou_loss(mask_1_2, label)
        #loss0 = structure_loss(mask_1_1, label, unc_1_1)#wbce(mask_1_1, label, unc_1_1) + iou_loss(mask_1_1, label)
        loss2 = F.binary_cross_entropy_with_logits(mask_1_4, label) + iou_loss(mask_1_4, label)
        loss1 = F.binary_cross_entropy_with_logits(mask_1_2, label) + iou_loss(mask_1_2, label)
        loss0 = F.binary_cross_entropy_with_logits(mask_1_1, label) + iou_loss(mask_1_1, label)
        
        loss = loss_weights[0] * loss0 + loss_weights[1] * loss1 + loss_weights[2] * loss2 + loss_weights[2] * loss2_ + loss_weights[3] * loss3 + loss_weights[4] * loss4

        #scloss3 = consistent_loss(sal_1_8,sal_1_16.detach()) * 0.0001
        #scloss2 = consistent_loss(sal_1_4,sal_1_8.detach()) * 0.0001

        loss = loss# + scloss3 + scloss2

        opt.zero_grad()
        loss.backward()
        opt.step()
        scheduler.step()
        epoch_total_loss += loss.cpu().data.item()
        epoch_loss0 += loss0.cpu().data.item()
        epoch_loss1 += loss1.cpu().data.item()
        epoch_loss2 += loss2.cpu().data.item()
        #epoch_loss3 += loss3.cpu().data.item()
        #epoch_loss4 += loss4.cpu().data.item()

        progress_bar.set_postfix(loss=f'{epoch_loss0/(i+1):.3f}')
    return epoch_loss0/l
        
def fit(model, train_dl, epochs=60, lr=1e-4,train_size = 384,save_dir = './loss.txt'):
    opt = get_opt(lr,model)
    scheduler = PolyLr(opt,gamma=0.9,minimum_lr=1.0e-07,max_iteration=len(train_dl)*epochs,warmup_iteration=12000)

    print('lr: '+str(lr))
    for epoch in range(epochs):
        #model.train()
        loss = train_one_epoch(epoch,epochs,model,opt,scheduler,train_dl,[train_size,train_size])
        fh = open(save_dir, 'a')
        if epoch == 0:
            fh.write('\n'+str(datetime.datetime.now())+'\n')
            fh.write('Start record.\n')
        fh.write(str(epoch+1) + ' current lr: ' + str(scheduler.get_lr()) + ' epoch_loss: ' + str(loss) + '\n')
        if (epoch+1)%10 == 0:
            if not os.path.exists('savepth/tmp/'):
                os.makedirs('savepth/tmp/')
            torch.save(model.state_dict(), 'savepth/tmp/'+str(epoch+1)+'.pth')
        if epoch+1 == epochs:
            fh.write(str(datetime.datetime.now())+'\n')
            fh.write('End record.\n')
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
    
def train(args):
    model = M3Net(dim=64,img_size=args.img_size,method=args.method,mode='train')
    if args.method == 'M3Net-R':
        model.encoder.load_state_dict(torch.load(args.pretrained_model), strict=False)
    elif args.method == 'M3Net-R2':
        model.encoder.load_state_dict(torch.load(args.pretrained_model, map_location='cpu'), strict=False)
    elif args.method == 'M3Net-S':
        model.encoder.load_state_dict(torch.load(args.pretrained_model, map_location='cpu')['model'], strict=False)

    print('Pre-trained weight loaded.')

    train_dataset = RGB_Dataset(root=args.data_root, sets=['DUTS-TR'],img_size=args.img_size,mode='train')
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle = True, 
                                               pin_memory=True,num_workers = 4,drop_last = True
                                               )
    
    model.cuda()
    model.train()
    print('Starting train.')
    fit(model,train_dl,args.train_epochs,args.lr,args.img_size,args.record)
    if not os.path.exists(args.save_model):
        os.makedirs(args.save_model)
    torch.save(model.state_dict(), args.save_model+args.method+'.pth')
    print('Saved as '+args.save_model+args.method+'.pth.')

if __name__ == '__main__':
    # Test
    print(type(str(datetime.datetime.now())))
    