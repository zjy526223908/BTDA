import sys
import random
import os
from PIL import Image
import tqdm
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from dataset.data_loader import get_loader,get_cluster_loader
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from dataset.data_loader import GetLoader
from torchvision import datasets
from torchvision import transforms
from models.model import Model,Domain_Classifier
import numpy as np
from tqdm import tqdm
import shutil
import torch.nn.functional as F
from vat import VATLoss
sys.path.append("../IDEC")
from idec import *

def make_npz_file(args):
    target_name_dict = {
        'mnist': 'mm_svhn_synth_usps', 
        'mnist_m': 'mnist_svhn_usps_synth', 
        'svhn': 'mnist_mm_usps_synth', 
        'synth': 'mnist_svhn_usps_mm', 
        'usps': 'mnist_svhn_synth_mm', 
    }
    x_train = np.array([])
    y_train = np.array([])
    
    x_train_tmp = np.array([])
    y_train_tmp = np.array([])
    count = 0
    
    print ("patch target set")
    fin = open("dataset/image_list/"+target_name_dict[args.source_name]+"_train.txt", "r")   
    for line in tqdm(fin):
        data = line.strip().split(" ")
        path = args.data_root+"/imgs/"+data[0]
        imgs = Image.open(path).convert('RGB')   
        imgs = imgs.resize((28, 28),Image.ANTIALIAS)
        img = np.asarray(imgs)
        x_train_tmp = np.append(x_train_tmp,img)
        y_train_tmp = np.append(y_train_tmp,0)
        count +=1
        
        if count%500==0:
            x_train = np.concatenate((x_train,x_train_tmp))
            y_train = np.concatenate((y_train,y_train_tmp))
            x_train_tmp = np.array([])
            y_train_tmp = np.array([])
        
    fin.close()
    x_train = np.concatenate((x_train,x_train_tmp))
    y_train = np.concatenate((y_train,y_train_tmp))
    x_train= x_train.reshape(count,28,28,3)
    y_train= y_train.reshape(count,-1)
    print (count)
    np.savez(args.image_npz_file, x_train=x_train,y_train=y_train)
    
    
def print_log(epoch, Classification_loss, Lent,Adversarial_DA_loss,Adversarial_DA_loss_Dst,\
    Lcf,Vmt,V_tilde_mt,tmp_accuracy,gamma_val, ploter, count):
    ploter.plot("Classification_loss", "train", count, Classification_loss)
    ploter.plot("Lent", "train", count, Lent)
    ploter.plot("Adversarial_DA_loss", "train", count, Adversarial_DA_loss)
    ploter.plot("Adversarial_DA_loss_Dst", "train", count, Adversarial_DA_loss_Dst)
    ploter.plot("Lcf", "train", count, Lcf)
    ploter.plot("Vmt", "train", count, Vmt)   
    ploter.plot("V_tilde_mt", "train", count, V_tilde_mt)
    ploter.plot("tmp_accuracy", "train", count, tmp_accuracy)
    ploter.plot("gamma_val", "train", count, gamma_val)
    
def test_model(t_loader_test,iter_count,args):
    model = Model(args.is_in,args.batch_size).cuda()
    model.load_state_dict(torch.load(args.snapshot_model_name))
    model.eval()

    correct = 0
    total = 0
    try:
        for (imgs, labels) in (t_loader_test):
            imgs = Variable(imgs.cuda())
            _,s_cls = model(imgs)
            #print s_cls
            s_cls = F.softmax(s_cls)
            s_cls = s_cls.data.cpu().numpy()     
            res = s_cls 
            
            pred = res.argmax(axis=1)
            labels = labels.numpy()
            #print labels
            correct += np.equal(labels, pred).sum()
            total +=labels.shape[0]
        current_accuracy = correct * 1.0 / total
        #print correct
        #print total
        print("Current accuracy is: {:.4f}".format(current_accuracy))
        
    except OSError:
        print("OSError")
        current_accuracy = 0
    except IOError:
        print("IOError")
        current_accuracy = 0
    except RuntimeError:
        print("RuntimeError")
        current_accuracy = 0  
    return current_accuracy



def test_model_equal_weight(args):
    target_name_list = ["mnist","mnist_m","usps","synth","svhn"]
    target_name_list.remove(args.source_name)
    # load data
    img_transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    count = 0
    total_acc = 0.0
    for target_name in target_name_list:
        #target test
        train_list = os.path.join("dataset/image_list/", target_name+'_test.txt')
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
        total_acc+=test_model(dataloader_target_test,0,args)
        count +=1
    ave_acc = total_acc*1.0/count
    
    
    return ave_acc


    
def update_teacher(t_loader_test,parser):
    print "saving feature concat image npz"
    args = parser.parse_args()
    model = Model(args.is_in,args.batch_size).cuda()
    model.load_state_dict(torch.load(args.snapshot_model_name))
    model.eval()
    count = 0
    teacher_feature = np.array([])
    tmp = np.array([])
    try:
        for (imgs, labels) in tqdm(t_loader_test):
            imgs = Variable(imgs.cuda())
            feature,s_cls = model(imgs)
            feature = feature.data.cpu().numpy() 
            s_cls = s_cls.data.cpu().numpy() 
            feature = feature.reshape(-1,64*8*8)
            
            teacher_feature_tmp = np.concatenate((feature,s_cls),axis=1) 
            tmp = np.append(tmp,teacher_feature_tmp)
            count+=1
            
            if count%50==0:
                teacher_feature = np.concatenate((teacher_feature,tmp))
                tmp = np.array([])
        teacher_feature = np.concatenate((teacher_feature,tmp))                
    except OSError:
        print("OSError")
    except IOError:
        print("IOError")
    except RuntimeError:
        print("RuntimeError")
    
    teacher_feature = teacher_feature.reshape(-1,64*8*8+10)
    npzfile = np.load(args.image_npz_file)
    x_train = npzfile['x_train']
    y_train = npzfile['y_train']
    x_train= x_train.reshape(-1,28*28*3)
    mix_feature = np.concatenate((x_train,teacher_feature),axis=1)
    np.savez(args.image_npz_update_file, x_train=mix_feature,y_train=y_train)
    
    print "updating meta"
    update_digit_meta_learner(parser)
    
    dataloader_cluster_label = get_cluster_loader(args)    
    return dataloader_cluster_label
    
def loss_entropy(input):
    loss = 0   
    '''for i in range(input.size()[0]):
        soft_max = F.softmax(input[i])
        loss += -1.0*torch.dot(soft_max,torch.log(soft_max))'''
    soft_max = F.softmax(input)  
    loss = -1.0*torch.dot(soft_max.view(-1),torch.log(soft_max+1e-20).view(-1))
    loss /=input.size()[0]    
    return loss

def two_loss_entropy(input1,labels):
    loss = 0
    '''for i in range(input1.size()[0]):
        soft_max = F.softmax(input1[i])
        soft_label = F.softmax(labels[i])      
        loss += -1.0*torch.dot(soft_label,torch.log(soft_max))'''
    soft_max = F.softmax(input1)
    soft_label = F.softmax(labels)    
    loss = -1.0*torch.dot(soft_label.view(-1),torch.log(soft_max+1e-20).view(-1))
    loss /=input1.size()[0]    
    return loss
    
def loss_log(input):
    loss = 0   
    '''for i in range(input.size()[0]):
        soft_max = F.softmax(input[i])
        loss += -1.0*torch.dot(soft_max,torch.log(soft_max))'''
    soft_max = F.softmax(input)  
    loss = -1.0*torch.log(soft_max+1e-20).view(-1).sum()
    loss /=(4.0*input.size()[0])  
    return loss
    
def loss_log_L2(input):
    loss = 0   
    '''for i in range(input.size()[0]):
        soft_max = F.softmax(input[i])
        loss += -1.0*torch.dot(soft_max,torch.log(soft_max))'''
    soft_max = F.softmax(input)  
    loss = torch.mean((soft_max - 0.25) ** 2) 
    return loss








    