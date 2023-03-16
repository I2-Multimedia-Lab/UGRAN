import argparse
from operator import gt
import os.path as osp
import os
import threading
from tkinter import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
#from SOD import SOD
from dataloader import get_loader
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
import transforms as trans
#from vst_token import ImageDepthNet
#from icon import ICON
from tqdm import tqdm
#from branch import branch
from M3Net import M3Net
def get_pred_dir(model, data_root = '/home/yy/datasets/', save_path = 'preds/',methods = 'DUTS+DUT-O+ECSSD+HKU-IS+PASCAL-S+SOD'):
    batch_size = 1
    test_paths = methods.split('+')
    for dataset_setname in test_paths:
        #print('get '+dataset_setname)
        img_root = dataset_setname + ''
        test_dataset = get_loader(img_root, data_root, 224, mode='test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

        progress_bar = tqdm(test_loader, desc=dataset_setname,ncols=140)
        for i,data_batch in enumerate(progress_bar):
            images, image_w, image_h, image_path = data_batch
            images = Variable(images.cuda())

            outputs_saliency = model(images)

            mask_1_1 = outputs_saliency[3]

            image_w, image_h = int(image_w[0]), int(image_h[0])

            output_s = torch.sigmoid(mask_1_1)

            transform = trans.Compose([
                transforms.ToPILImage(),
                trans.Scale((image_w, image_h))
            ])
            #output_s = output_s.data.cpu().squeeze(0)
            thread = threading.Thread(target = save_p,args = (output_s.shape[0],output_s,image_w,image_h,image_path,dataset_setname,save_path))
            thread.start()
        '''
        img_files = os.listdir(img_root)
        for i in img_files:
            img = cv2.imread(img_root+i)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = cv2.resize(img,(224,224))
            img = np.asarray(img)
            img = img.astype(np.float32)
            imgs = torch.sigmoid(model(torch.tensor(img.transpose(2,0,1)).view(-1,3,224,224).cuda())[0]).cpu().detach().numpy()[0][0]
            imgs = imgs*255
            cv2.imwrite("Evaluation/sals/"+dataset_setname+'/'+i[:-4]+'.png',imgs)
        '''

def save_p(size,outputs,image_w,image_h,image_path,dataset_setname,save_path):
    transform = trans.Compose([
                transforms.ToPILImage(),
                trans.Scale((image_w, image_h))
            ])
    for ii in range(0,size):

        output_si = outputs[ii]
        output_si = transform(output_si)

        #dataset = img_root.split('/')[0]
        filename = image_path[ii].split('/')[-1].split('.')[0]

        # save saliency maps
        save_test_path = save_path+dataset_setname+'/'
        if not os.path.exists(save_test_path):
            os.makedirs(save_test_path)
        output_si.save(os.path.join(save_test_path, filename + '.png'))

def testing(args):
    print('Starting test.')
    model = M3Net(embed_dim=384,dim=96,img_size=224,method=args.method)
    model.cuda()
    model.load_state_dict(torch.load(args.save_model+args.method+'.pth'))
    print('Loaded from '+args.save_model+args.method+'.pth.')
    model.eval()
    get_pred_dir(model,data_root=args.data_root,save_path=args.save_test,methods=args.test_methods)
    print('Predictions are saved at '+args.save_test+'.')
