import sys
import random
import os
from PIL import Image
import tqdm
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from dataset.data_loader import get_loader,get_cluster_loader
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from models.model import *
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
import shutil
import torch.nn.functional as F
from vat import VATLoss
from dataset.data_loader import *
sys.path.append("../IDEC")
from idec import *

def make_npz_file(args):
    target_name_dict = {
        'A': 'RPC', 
        'C': 'RPA', 
        'P': 'RCA', 
        'R': 'PCA', 
        'amazon': 'DW', 
        'webcam': 'AD', 
        'dslr': 'AW', 
    }
    x_train = np.array([])
    y_train = np.array([])
    
    x_train_tmp = np.array([])
    y_train_tmp = np.array([])
    count = 0
    
    print ("patch target traning set")
    if args.dataset_name == "OfficeHome":
        fin = open("dataset/"+args.dataset_name+"/"+target_name_dict[args.source_name]+"_domain_List.txt", "r")   
    else:
        fin = open("dataset/"+args.dataset_name+"/"+target_name_dict[args.source_name]+"_domain_list_equal.txt", "r")   
        
    for line in tqdm(fin):
        data = line.strip().split(" ")
        path = args.data_root+data[0]
        imgs = Image.open(path).convert('RGB') 
        imgs = imgs.resize((64, 64),Image.ANTIALIAS)
        img = np.asarray(imgs)
        x_train_tmp = np.append(x_train_tmp,img)
        
        y_train_tmp = np.append(y_train_tmp,int(data[1]))
        count +=1
        
        if count%100==0:
            x_train = np.concatenate((x_train,x_train_tmp))
            y_train = np.concatenate((y_train,y_train_tmp))
            x_train_tmp = np.array([])
            y_train_tmp = np.array([])
        
    fin.close()
    x_train = np.concatenate((x_train,x_train_tmp))
    y_train = np.concatenate((y_train,y_train_tmp))
    x_train= x_train.reshape(count,64,64,3)
    y_train= y_train.reshape(count,-1)
    print (count)
    np.savez(args.image_npz_file, x_train=x_train,y_train=y_train)
    
   

def make_test_npz_file(dataloader_target_test,args):
    x_train = np.array([])
    y_train = np.array([])
    
    x_train_tmp = np.array([])
    y_train_tmp = np.array([])
    count = 0
    print ("patch target testing set")
    for (imgs, labels) in tqdm(dataloader_target_test):
        imgs = imgs.data.cpu().numpy()   
        labels = labels.data.cpu().numpy()
        x_train_tmp = np.append(x_train_tmp,imgs)
        y_train_tmp = np.append(y_train_tmp,labels)
        count +=labels.shape[0]
        if count%50==0:
            x_train = np.concatenate((x_train,x_train_tmp))
            y_train = np.concatenate((y_train,y_train_tmp))
            x_train_tmp = np.array([])
            y_train_tmp = np.array([])
    x_train = np.concatenate((x_train,x_train_tmp))
    y_train = np.concatenate((y_train,y_train_tmp)) 
    print (x_train.shape)
    print (y_train.shape) 
    print (count) 
    x_train= x_train.reshape(count,3,227,227)
    y_train= y_train.reshape(count,-1)    
    print (x_train.shape)
    print (y_train.shape)    
    np.savez(args.image_test_npz_file, x_train=x_train,y_train=y_train)
   
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


class Test_Dataset(Dataset):

    def __init__(self,path):
        self.x, self.y = load_all(path)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))
                        
def load_all(path):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    y_train = y_train.reshape(-1)
    y_train = y_train.astype(np.int32)
    x_train = x_train.astype(np.float32)
    f.close()
    return x_train, y_train

    
def test_model(t_loader_test,iter_count,args): 
    if args.dataset_name == "OfficeHome":
        test_net = Res_Model_OfficeHome().cuda()
    else:
        test_net = Res_Model_Office31().cuda() 
    test_net.load_state_dict(torch.load(args.snapshot_model_name))
    test_net.eval()
    correct = 0
    total = 0
    try:
        for (imgs, labels,_) in tqdm(t_loader_test):
            imgs = Variable(imgs.cuda())
            s_cls,_ ,_,_= test_net(imgs,0)
            s_cls = F.softmax(s_cls)
            s_cls = s_cls.data.cpu().numpy()     
            res = s_cls 
            
            pred = res.argmax(axis=1)
            labels = labels.numpy()   
            correct += np.equal(labels, pred).sum()
            total +=labels.shape[0]
        current_accuracy = correct * 1.0 / total
        print("Current accuracy is: {:.4f}%".format(current_accuracy*100.0))
        
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


def test_model_single(t_loader_test,iter_count,args): 
    if args.dataset_name == "OfficeHome":
        test_net = Res_Model_OfficeHome().cuda()
    else:
        test_net = Res_Model_Office31().cuda()  
    test_net.load_state_dict(torch.load(args.snapshot_max_accuracy_model_name))
    test_net.eval()
    correct = 0
    total = 0
    try:
        for (imgs, labels) in tqdm(t_loader_test):
            imgs = Variable(imgs.cuda())
            s_cls,_ ,_,_= test_net(imgs,0)
            s_cls = F.softmax(s_cls)
            s_cls = s_cls.data.cpu().numpy()     
            res = s_cls 
            
            pred = res.argmax(axis=1)
            labels = labels.numpy()   
            correct += np.equal(labels, pred).sum()
            total +=labels.shape[0]
        current_accuracy = correct * 1.0 / total
        print("Current accuracy is: {:.4f}%".format(current_accuracy*100.0))
        
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
    if args.dataset_name == "OfficeHome":
        target_name_list = ["A","C","R","P"]
    else:
        target_name_list = ["amazon","webcam","dslr"]
    target_name_list.remove(args.source_name)
    # load data

    count = 0
    total_acc = 0.0
    for target_name in target_name_list:
        #target test
        if args.dataset_name == "OfficeHome":
            test_list = "dataset/"+args.dataset_name+"/"+target_name+"List.txt"
        else:
            test_list = "dataset/"+args.dataset_name+"/"+target_name+".txt"
        test_set = OfficeImage(args.data_root, test_list ,split="test")
        cluster_label_raw = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size,
            shuffle=False, num_workers=8)
            
        total_acc+=test_model_single(cluster_label_raw,0,args)
        count +=1
    ave_acc = total_acc*1.0/count
    return ave_acc    

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
    
def update_teacher(dataloader_no_shuffle,parser):
    print "saving feature concat image npz"
    args = parser.parse_args()
    if args.dataset_name == "OfficeHome":
        model = Res_Model_OfficeHome().cuda()
    else:
        model = Res_Model_Office31().cuda() 
    model.load_state_dict(torch.load(args.snapshot_model_name))
    model.eval()
    count = 0
    teacher_feature = np.array([])
    tmp = np.array([])
    try:
        for (imgs, labels) in tqdm(dataloader_no_shuffle):
            imgs = Variable(imgs.cuda())
            s_cls,_ ,_,feature = model(imgs,0)
            feature = feature.data.cpu().numpy() 
            s_cls = s_cls.data.cpu().numpy() 
            feature = feature.reshape(-1,4096*2)
            
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
    
    if args.dataset_name == "OfficeHome":
        teacher_feature = teacher_feature.reshape(-1,4096*2+65)
    else:
        teacher_feature = teacher_feature.reshape(-1,4096*2+31)
    npzfile = np.load(args.image_npz_file)
    x_train = npzfile['x_train']
    y_train = npzfile['y_train']
    x_train= x_train.reshape(-1,64*64*3)
    mix_feature = np.concatenate((x_train,teacher_feature),axis=1)
    np.savez(args.image_npz_update_file, x_train=mix_feature,y_train=y_train)
    
    print "updating meta"
    update_office_meta_learner(parser)

    dataloader_cluster_label = get_cluster_loader(args)    
    return dataloader_cluster_label   
    
    