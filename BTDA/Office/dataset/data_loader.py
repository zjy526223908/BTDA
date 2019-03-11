import torch.utils.data as data
from PIL import Image
import os
import sys
import random
import os
import torch.nn as nn
import torch
import math
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
import numpy as np
import tqdm
import torch.nn.functional as F
import shutil
import sys

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def make_dataset(root, label):
    images = []
    labeltxt = open(label)
    for line in labeltxt:
        data = line.strip().split(' ')
        #print data
        if is_image_file(data[0]):
            path = os.path.join(root, data[0])
        gt = int(data[1])
        item = (path, gt)
        images.append(item)
    return images


class OfficeImage(data.Dataset):
    def __init__(self, root, label, split="train", transform=None):
        imgs = make_dataset(root, label)
        self.root = root
        self.label = label
        self.split = split
        self.imgs = imgs
        self.transform = transform
        self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
 
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert("RGB")
        
        img = img.resize((256, 256), Image.BILINEAR)

        if self.split == "train":
            w, h = img.size
            tw, th = (227, 227)
            x1 = np.random.randint(0, w - tw)
            y1 = np.random.randint(0, h - th)
            img = img.crop((x1, y1, x1 + tw, y1 + th))
        if self.split == "test":
            img = img.crop((15, 15, 242, 242))

        img = np.array(img, dtype=np.float32)
        img = img[:, :, ::-1]
        img = img - self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
            
        return img, target

    def __len__(self):
        return len(self.imgs)
        
        
def get_loader(args):
    target_name_dict = {
        'A': 'RPC', 
        'C': 'RPA', 
        'P': 'RCA', 
        'R': 'PCA', 
        'amazon': 'DW', 
        'webcam': 'AD', 
        'dslr': 'AW', 
    }
    
    # load data 
    s_root = args.data_root
    t_root = args.data_root    
    if args.dataset_name == "OfficeHome": 
        s_label = "dataset/"+args.dataset_name+"/"+args.source_name+"List.txt"
        t_label = "dataset/"+args.dataset_name+"/"+target_name_dict[args.source_name]+"List.txt"
    else:
        s_label = "dataset/"+args.dataset_name+"/"+args.source_name+".txt"
        t_label = "dataset/"+args.dataset_name+"/"+target_name_dict[args.source_name]+"List.txt"
    #source train   
    s_set = OfficeImage(s_root, s_label ,split="train")
    s_train_loader_raw = torch.utils.data.DataLoader(s_set, batch_size=args.batch_size,
        shuffle=True, num_workers=8) 
        
    #target test     
    t_set = OfficeImage(t_root, t_label,split="test")
    t_test_loader_raw = torch.utils.data.DataLoader(t_set, batch_size=args.batch_size,
        shuffle=False, num_workers=8)    
        
    return s_train_loader_raw,t_test_loader_raw


def get_cluster_loader(args):
    #cluster train      
    train_list = os.path.join(args.update_list_file, 'cluster_label.txt') 
    cluster_set = OfficeImage(args.data_root, train_list ,split="train")
    cluster_label_raw = torch.utils.data.DataLoader(cluster_set, batch_size=args.batch_size,
        shuffle=True, num_workers=8)   
    return cluster_label_raw   
    
    
def get_noshuffle_loader(args):
    target_name_dict = {
        'A': 'RPC', 
        'C': 'RPA', 
        'P': 'RCA', 
        'R': 'PCA', 
        'amazon': 'DW', 
        'webcam': 'AD', 
        'dslr': 'AW', 
    }
    # load data
    img_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    #cluster train      
    if args.dataset_name == "OfficeHome":
        train_list = os.path.join("dataset/OfficeHome/", target_name_dict[args.source_name]+'_domain_List.txt')
    else:
        train_list = os.path.join("dataset/Office31/", target_name_dict[args.source_name]+'_domain_list_equal.txt')
    domain_set = OfficeImage(args.data_root, train_list ,split="train")
    domain_label_raw = torch.utils.data.DataLoader(domain_set, batch_size=args.batch_size,
        shuffle=False, num_workers=8)
    return domain_label_raw    
    
def get_provided_cluster_loader(args):
    target_name_dict = {
        'A': 'RPC', 
        'C': 'RPA', 
        'P': 'RCA', 
        'R': 'PCA', 
        'amazon': 'DW', 
        'webcam': 'AD', 
        'dslr': 'AW', 
    }
    # load data
    img_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    #cluster train 
    if args.dataset_name == "OfficeHome":
        train_list = os.path.join("dataset/OfficeHome/", target_name_dict[args.source_name]+'_cluster_List.txt')
    else:
        train_list = os.path.join("dataset/Office31/", target_name_dict[args.source_name]+'_domain_cluster_list_equal.txt')
    cluster_set = OfficeImage(args.data_root, train_list ,split="train")
    cluster_label_raw = torch.utils.data.DataLoader(cluster_set, batch_size=args.batch_size,
        shuffle=True, num_workers=8)   

    return cluster_label_raw 