import torch
import torch.nn.functional as F
from tqdm import tqdm
from M3Net import M3Net
from dataloader import get_loader
import os
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

def train_one_epoch(epoch,epochs,model,opt,train_dl):
    epoch_total_loss = 0
    epoch_loss1 = 0
    epoch_loss2 = 0
    epoch_loss3 = 0
    epoch_loss4 = 0

    loss_weights = [1, 0.8, 0.5, 0.5, 0.5]
    l = 0

    progress_bar = tqdm(train_dl, desc='Epoch[{:03d}/{:03d}]'.format(epoch+1, epochs),ncols=140)
    for i, data_batch in enumerate(progress_bar):

        l = l+1

        images, label, label_1_16, label_1_8, label_1_4 = data_batch
        images, label = images.cuda(non_blocking=True), label.cuda(non_blocking=True)
        label_1_16, label_1_8, label_1_4 = label_1_16.cuda(), label_1_8.cuda(), label_1_4.cuda()

        out2, out3, out4, out5 = model(images)
        
        loss4  = F.binary_cross_entropy_with_logits(out2, label_1_16) + iou_loss(out2, label_1_16)
        loss3  = F.binary_cross_entropy_with_logits(out3, label_1_8) + iou_loss(out3, label_1_8)
        loss2  = F.binary_cross_entropy_with_logits(out4, label_1_4) + iou_loss(out4, label_1_4)
        loss1  = F.binary_cross_entropy_with_logits(out5, label) + iou_loss(out5, label)

        loss = loss_weights[0] * loss1 + loss_weights[1] * loss2 + loss_weights[2] * loss3 + loss_weights[3] * loss4

        opt.zero_grad()
        loss.backward()
        opt.step()

        epoch_total_loss += loss.cpu().data.item()
        epoch_loss1 += loss1.cpu().data.item()
        epoch_loss2 += loss2.cpu().data.item()
        epoch_loss3 += loss3.cpu().data.item()
        epoch_loss4 += loss4.cpu().data.item()

        progress_bar.set_postfix(loss=f'{epoch_loss1/(i+1):.3f}')
    return epoch_loss1/l
        
def fit(model, train_dl, epochs=[100,20], lr=1e-4):
    step = len(epochs)
    save_dir = './loss.txt'
    for st in range(step):
        opt = get_opt(lr,model)
        print('Starting train step {}.'.format(st+1))
        print('lr: '+str(lr))
        for epoch in range(epochs[st]):
            #model.train()
            loss = train_one_epoch(epoch,epochs[st],model,opt,train_dl)
            fh = open(save_dir, 'a')
            if(epoch == 0):
                fh.write('Step: ' + str(st+1) + ', current lr: ' + str(lr) + '\n')
            fh.write(str(epoch+1) + ' epoch_loss: ' + str(loss) + '\n')
            fh.close()
        lr = lr/5

def get_opt(lr,model):
    
    base_params = [params for name, params in model.named_parameters() if ("encoder" in name)]
    other_params = [params for name, params in model.named_parameters() if ("encoder" not in name)]
    params = [{'params': base_params, 'lr': lr*0.1},
          {'params': other_params, 'lr': lr}
         ]
         
    opt = torch.optim.Adam(params, lr)

    return opt

def training(args):
    if args.method == 'M3Net-S':
        model = M3Net(embed_dim=384,dim=96,img_size=224,method=args.method)
        model.encoder.load_state_dict(torch.load('./pretrained_model/swin_small_patch4_window7_224.pth')['model'])
    elif args.method == 'M3Net-R':
        model = M3Net(embed_dim=384,dim=96,img_size=224,method=args.method)
        model.encoder.load_state_dict(torch.load('./pretrained_model/ResNet50.pth'))
    elif args.method == 'M3Net-T':
        model = M3Net(embed_dim=384,dim=64,img_size=224,method=args.method)
        model.encoder.load_state_dict(torch.load('/pretrained_model/T2T_ViTt_14.pth.tar')['state_dict_ema'])
    print('Pre-trained weight loaded.')

    train_dataset = get_loader(args.trainset, args.data_root, 224, mode='train')
    train_dl = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle = True, 
                                               pin_memory=True,num_workers = 2
                                               )
    
    model.cuda()
    model.train()
    print('Starting train.')
    fit(model,train_dl,[args.step1epochs,args.step2epochs],args.lr)
    if not os.path.exists(args.save_model):
        os.makedirs(args.save_model)
    torch.save(model.state_dict(), args.save_model+args.method+'.pth')
    print('Saved as '+args.save_model+args.method+'.pth.')
