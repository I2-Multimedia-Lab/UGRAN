# --------------------------------------------------------
# InSPyReNet
# Copyright (c) 2021 Taehun Kim
# Licensed under The MIT License 
# https://github.com/plemeri/InSPyReNet
# --------------------------------------------------------

import os
import sys
import re
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from PIL import Image

filepath = os.path.split(__file__)[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from data.custom_transforms import *

Image.MAX_IMAGE_PIXELS = None

def get_transform(img_size=384,mode = 'train'):
    comp = []
    if mode == 'train':
        # Data enhancement applied
        comp.append(static_resize(size = [img_size,img_size]))
        comp.append(random_scale_crop(range = [0.75,1.25]))
        comp.append(random_flip(lr = True, ud = False))
        comp.append(random_rotate(range = [-10,10]))
        comp.append(random_image_enhance(methods=  ['contrast', 'sharpness', 'brightness']))
        comp.append(tonumpy())
        comp.append(normalize(mean= [0.485, 0.456, 0.406] ,std= [0.229, 0.224, 0.225]))
        comp.append(totensor())
    else:
        comp.append(static_resize(size = [img_size,img_size]))
        comp.append(tonumpy())
        comp.append(normalize(mean= [0.485, 0.456, 0.406] ,std= [0.229, 0.224, 0.225]))
        comp.append(totensor())
    return transforms.Compose(comp)

class RGB_Dataset(Dataset):
    def __init__(self, root, sets, img_size, mode):
        self.images, self.gts = [], []
        
        for set in sets:
            image_root, gt_root = os.path.join(root, set, 'RGB'), os.path.join(root, set, 'GT')

            images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.lower().endswith(('.jpg', '.png'))]
            images = sort(images)
            
            gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.lower().endswith(('.jpg', '.png'))]
            gts = sort(gts)
            
            self.images.extend(images)
            self.gts.extend(gts)
        
        self.filter_files()
        
        self.size = len(self.images)
        self.transform = get_transform(img_size,mode)
        
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        gt = Image.open(self.gts[index]).convert('L')
        shape = gt.size
        
        name = self.images[index].split(os.sep)[-1]
        name = os.path.splitext(name)[0]
            
        sample = {'image': image, 'gt': gt, 'name': name, 'shape': shape}

        sample = self.transform(sample)
        return sample

    def filter_files(self):
        assert len(self.images) == len(self.gts)
        images, gts = [], []
        for img_path, gt_path in zip(self.images, self.gts):
            img, gt = Image.open(img_path), Image.open(gt_path)
            if img.size == gt.size:
                images.append(img_path)
                gts.append(gt_path)
        self.images, self.gts = images, gts

    def __len__(self):
        return self.size
    
class ImageLoader:
    def __init__(self, root, tfs):
        if os.path.isdir(root):
            self.images = [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            self.images = sort(self.images)
        elif os.path.isfile(root):
            self.images = [root]
        self.size = len(self.images)
        self.transform = get_transform(tfs)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index == self.size:
            raise StopIteration
        image = Image.open(self.images[self.index]).convert('RGB')
        shape = image.size[::-1]
        name = self.images[self.index].split(os.sep)[-1]
        name = os.path.splitext(name)[0]
            
        sample = {'image': image, 'name': name, 'shape': shape, 'original': image}
        sample = self.transform(sample)
        sample['image'] = sample['image'].unsqueeze(0)
        if 'image_resized' in sample.keys():
            sample['image_resized'] = sample['image_resized'].unsqueeze(0)
        
        self.index += 1
        return sample

    def __len__(self):
        return self.size
    

def sort(x):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(x, key=alphanum_key)