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

class GetLoader(data.Dataset):
    def __init__(self, data_root, data_list, transform=None):
        self.root = data_root
        self.transform = transform

        f = open(data_list, 'r')
        data_list = f.readlines()
        f.close()

        self.n_data = len(data_list)

        self.img_paths = []
        self.img_labels = []

        for data in data_list:
            self.img_paths.append(data[:-3])
            self.img_labels.append(data[-2])

    def __getitem__(self, item):
        img_paths, labels = self.img_paths[item], self.img_labels[item]
        imgs = Image.open(os.path.join(self.root, img_paths)).convert('RGB')

        if self.transform is not None:
            imgs = self.transform(imgs)
            labels = int(labels)

        return imgs, labels

    def __len__(self):
        return self.n_data   
        
        
def get_loader(args):
    target_name_dict = {
        'mnist': 'mm_svhn_synth_usps', 
        'mnist_m': 'mnist_svhn_usps_synth', 
        'svhn': 'mnist_mm_usps_synth', 
        'synth': 'mnist_svhn_usps_mm', 
        'usps': 'mnist_svhn_synth_mm', 
    }

    # load data
    img_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    #source train    
    train_list = os.path.join("dataset/image_list/", args.source_name+'_train.txt')
    dataset_source_train = GetLoader(
        data_root = os.path.join(args.data_root, 'imgs'),
        data_list = train_list,
        transform = img_transform
    )
    dataloader_source_train = torch.utils.data.DataLoader(
        dataset = dataset_source_train,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers=8)
    #target train
    train_list = os.path.join("dataset/image_list/", target_name_dict[args.source_name]+'_train.txt')
    dataset_target_train = GetLoader(
        data_root = os.path.join(args.data_root, 'imgs'),
        data_list = train_list,
        transform = img_transform
    )
    dataloader_target_train = torch.utils.data.DataLoader(
        dataset=dataset_target_train,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers=8)
    dataloader_target_train_noshuffle = torch.utils.data.DataLoader(
        dataset = dataset_target_train,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = 8)    
    #source test      
    train_list = os.path.join("dataset/image_list/", args.source_name+'_test.txt')
    dataset_source_test = GetLoader(
        data_root = os.path.join(args.data_root, 'imgs'),
        data_list = train_list,
        transform = img_transform
    )
    dataloader_source_test = torch.utils.data.DataLoader(
        dataset = dataset_source_test,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = 8)
    #target test
    train_list = os.path.join("dataset/image_list/", target_name_dict[args.source_name]+'_test.txt')
    dataset_target_test = GetLoader(
        data_root = os.path.join(args.data_root, 'imgs'),
        data_list = train_list,
        transform = img_transform
    )
    dataloader_target_test = torch.utils.data.DataLoader(
        dataset = dataset_target_test,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = 8)

    return dataloader_source_train,dataloader_source_test,dataloader_target_train,dataloader_target_train_noshuffle,dataloader_target_test


def get_cluster_loader(args):
    # load data
    img_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    #source train    
    train_list = os.path.join(args.update_list_file, 'cluster_label.txt')
    dataset_target_cluster = GetLoader(
        data_root=os.path.join(args.data_root, 'imgs'),
        data_list=train_list,
        transform=img_transform
    )
    cluster_label_raw = torch.utils.data.DataLoader(
        dataset=dataset_target_cluster,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8)   

    return cluster_label_raw   
    
def get_provided_cluster_loader(args):
    target_name_dict = {
        'mnist': 'mm_svhn_synth_usps', 
        'mnist_m': 'mnist_svhn_usps_synth', 
        'svhn': 'mnist_mm_usps_synth', 
        'synth': 'mnist_svhn_usps_mm', 
        'usps': 'mnist_svhn_synth_mm', 
    }
    # load data
    img_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    #source train    
    train_list = os.path.join("dataset/initialize_cluster_list/", target_name_dict[args.source_name]+'_cluster_train.txt')
    dataset_target_cluster = GetLoader(
        data_root=os.path.join(args.data_root, 'imgs'),
        data_list=train_list,
        transform=img_transform
    )
    cluster_label_raw = torch.utils.data.DataLoader(
        dataset=dataset_target_cluster,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8)   

    return cluster_label_raw       