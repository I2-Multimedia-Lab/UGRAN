#import pathlib
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
import random
import os

def load_list(dataset_name, data_root):

    images = []
    labels = []

    datasets = dataset_name.split('+')

    for dataset in datasets:
        img_root = data_root + '/' + dataset+ '/imgs/'
        img_files = os.listdir(img_root)

        for img in img_files:

            images.append(img_root + img[:-4]+'.jpg')
            labels.append(img_root.replace('/imgs/', '/gt/') + img[:-4]+'.png')

    return images, labels


def load_test_list(test_path, data_root):

    images = []

    img_root = data_root + test_path + '/imgs/'
    img_files = os.listdir(img_root)

    if '/HKU-IS/' in img_root:
        ext = '.png'
    else:
        ext = '.jpg'
        
    for img in img_files:
        images.append(img_root + img[:-4] + ext)

    return images


class ImageData(data.Dataset):
    def __init__(self, dataset_list, data_root, transform, mode, img_size=None, scale_size=None, t_transform=None, label_1_16_transform=None, label_1_8_transform=None, label_1_4_transform=None, label_1_2_transform=None):

        if mode == 'train':
            self.image_path, self.label_path = load_list(dataset_list, data_root)
        else:
            self.image_path = load_test_list(dataset_list, data_root)

        self.transform = transform
        self.t_transform = t_transform
        self.label_1_16_transform = label_1_16_transform
        self.label_1_8_transform = label_1_8_transform
        self.label_1_4_transform = label_1_4_transform
        self.label_1_2_transform = label_1_2_transform
        self.mode = mode
        self.img_size = img_size
        self.scale_size = scale_size

    def __getitem__(self, item):
        fn = self.image_path[item].split('/')

        filename = fn[-1]

        image = Image.open(self.image_path[item]).convert('RGB')
        image_w, image_h = int(image.size[0]), int(image.size[1])

        if self.mode == 'train':

            label = Image.open(self.label_path[item]).convert('L')
            random_size = self.scale_size

            new_img = transforms.Resize((random_size, random_size))(image)
            new_label = transforms.Resize((random_size, random_size), interpolation=Image.NEAREST)(label)

            # random crop
            w, h = new_img.size
            if w != self.img_size and h != self.img_size:
                x1 = random.randint(0, w - self.img_size)
                y1 = random.randint(0, h - self.img_size)
                new_img = new_img.crop((x1, y1, x1 + self.img_size, y1 + self.img_size))
                new_label = new_label.crop((x1, y1, x1 + self.img_size, y1 + self.img_size))

            # random flip
            if random.random() < 0.5:
                new_img = new_img.transpose(Image.FLIP_LEFT_RIGHT)
                new_label = new_label.transpose(Image.FLIP_LEFT_RIGHT)

            new_img = self.transform(new_img)

            label_1_16 = self.label_1_16_transform(new_label)
            label_1_8 = self.label_1_8_transform(new_label)
            label_1_4 = self.label_1_4_transform(new_label)
            label_1_2 = self.label_1_2_transform(new_label)
            label_224 = self.t_transform(new_label)


            return new_img, label_224, label_1_16, label_1_8, label_1_4, label_1_2
        else:

            image = self.transform(image)

            return image, image_w, image_h, self.image_path[item]

    def __len__(self):
        return len(self.image_path)


def get_loader(dataset_list, data_root, img_size, mode='train'):

    if mode == 'train':

        transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        t_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        label_1_16_transform = transforms.Compose([
            transforms.Resize((img_size // 16, img_size // 16), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_1_8_transform = transforms.Compose([
            transforms.Resize((img_size//8, img_size//8), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_1_4_transform = transforms.Compose([
            transforms.Resize((img_size//4, img_size//4), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        label_1_2_transform = transforms.Compose([
            transforms.Resize((img_size//2, img_size//2), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])
        if img_size == 224:
            scale_size = 256
        else:
            scale_size = 512
    else:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),  # 处理的是Tensor
        ])

    if mode == 'train':
        dataset = ImageData(dataset_list, data_root, transform, mode, img_size, scale_size, t_transform, label_1_16_transform, label_1_8_transform, label_1_4_transform, label_1_2_transform)
    else:
        dataset = ImageData(dataset_list, data_root, transform, mode)

    return dataset