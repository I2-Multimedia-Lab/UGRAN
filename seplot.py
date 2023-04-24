from xxSOD import M3Net
import torch
from matplotlib import pyplot as plt
from data.dataloader import RGB_Dataset
import numpy as np
model = M3Net(dim=64,img_size=384,method='M3Net-R')
model.cuda()
model.eval()
model.load_state_dict(torch.load('savepth/M3Net-R_MFE_SE.pth'))
test_dataset = RGB_Dataset('/mnt/ssd/yy/datasets/', ['DUTS-TE'], 384,'test')
test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle = True, 
                                               pin_memory=True,num_workers = 4)

l = []
for i,data_batch in enumerate(test_dl):
    images = data_batch['image']
    image_w,image_h = data_batch['shape']
    image_path = data_batch['name']
    images = images.cuda()

    y,w = model(images)
    #print(w.shape)
    w = w.squeeze(2).squeeze(2).squeeze(0)
    l.append(w.clone().detach().cpu().numpy())
    if(i == 200):
        break
cnt = 200
l1 = []
l2 = []
l3 = []
l4 = []
l5 = []
threshold=512
for i in range(0,cnt):
    l1.append(np.average(l[i][0:threshold]))
    l2.append(np.average(l[i][threshold:2*threshold]))
    l3.append(np.average(l[i][2*threshold:3*threshold]))
    l4.append(np.average(l[i][3*threshold:4*threshold]))
    l5.append(np.average(l[i][4*threshold:5*threshold]))
ll = np.concatenate([l1,l2,l3,l4,l5],axis=0)
x = np.arange(1,1001)
size = np.ones(1001)
plt.scatter(x,ll,sizes=size)
plt.savefig('se55.png')