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
from models.model import *
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from vat import VATLoss
from utils import LinePlotter
from dataset.data_loader import get_loader,get_cluster_loader,get_provided_cluster_loader,get_noshuffle_loader
from alex_function import *
import shutil
import time
time = str(time.strftime('%Y_%m_%d %H:%M:%S',time.localtime(time.time())))
import argparse
sys.path.append("../IDEC")
from idec import *

#2: initialize hyper-parameter and environment
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", help="set dataset name, Office31 or OfficeHome", default="OfficeHome")
parser.add_argument("--model_name", help="model name", default="alexnet")
parser.add_argument("--source_name", help="set source domain, form Office31:(amazon dslr webcam) OfficeHome:(A,C,P,R)", default="A")
parser.add_argument("--batch_size", help="set batch size", type=int, default=32)
parser.add_argument("--update_meta_iter", help="set update meta iter number", type=int, default=2000)
parser.add_argument("--test_iter", help="set test iter number", type=int, default=40)
parser.add_argument("--snapshot", help="set which folder to save model and result", default="snapshot/")
parser.add_argument("--lr", help="set learning rate", type=int, default=1e-3)
parser.add_argument("--gpu_id", help="set gpu id", type=int, default=0)
parser.add_argument("--is_in", help="whether use ", type=int, default=0)
parser.add_argument("--image_size", help="set input image size", type=int, default=227)
parser.add_argument("--max_iter", help="set max iter times", type=int, default = 40000)
parser.add_argument("--beta_val", help="set beta value", type=int, default= 0.01 )
parser.add_argument("--lambda_val", help="set lambda value", type=int, default= 0.1)
parser.add_argument("--gamma_val", help="set gamma value", type=int, default= 0.01)
parser.add_argument("--rho_val", help="set rho value", type=int, default= 0.01)
parser.add_argument("--is_initialize", help="whether initialize cluster labels or use provided cluster labels", type=int,default=0)
parser.add_argument("--data_root", help="set dataset file", default=('dataset/'+parser.parse_args().dataset_name+'/imgs'))
parser.add_argument("--task_name", help="generate this task's name", \
    default= parser.parse_args().source_name+'_2_all')
parser.add_argument("--snapshot_file", help="generate folder to save models and result", \
    default='snapshot/'+'alexnet_'+parser.parse_args().source_name+'_2_other_'+time )
parser.add_argument("--update_list_file", \
    default= parser.parse_args().snapshot_file+'/update_list/' )
parser.add_argument("--snapshot_model_file", \
    default= parser.parse_args().snapshot_file+'/model/' )
parser.add_argument("--snapshot_model_name", \
    default= parser.parse_args().snapshot_model_file+parser.parse_args().task_name+'_current_model.pth')
parser.add_argument("--snapshot_max_accuracy_model_name", \
    default= parser.parse_args().snapshot_model_file+parser.parse_args().task_name+'_max_accuracy_model.pth')
parser.add_argument("--result_file_folder",  \
    default= parser.parse_args().snapshot_file+'/result/')
parser.add_argument("--result_file_name", \
    default= parser.parse_args().snapshot_file+'/result/result.txt')
parser.add_argument("--image_npz_file", help="pack images to npz flie", \
    default= 'dataset/image_npz/'+parser.parse_args().source_name+'.npz' )   
parser.add_argument("--image_test_npz_file", help="pack test images to npz flie to boost test", \
    default= 'dataset/image_npz/'+parser.parse_args().source_name+'_test.npz' )     
parser.add_argument("--image_npz_update_file", \
    default= parser.parse_args().update_list_file+parser.parse_args().source_name+'_alexnet_update.npz' )    
parser.add_argument("--max_accuracy", help="max accuracy on test data", default= 0.0 )
parser.add_argument("--max_accuracy_iter", help="max accuracy's iter num", default= 0 )
parser.add_argument("--tmp_accuracy", help="current accuracy on test data", default= 0.0 )
args = parser.parse_args()


os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
ploter = LinePlotter(args)
loss_class = torch.nn.CrossEntropyLoss().cuda()
loss_domain = torch.nn.BCEWithLogitsLoss().cuda()


 

cuda = True
cudnn.benchmark = True
manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)
 
# creat snapshot


if os.path.exists(args.snapshot_file)==False:
    os.makedirs(args.snapshot_file)
    os.makedirs(args.update_list_file)
    os.makedirs(args.snapshot_model_file)
    os.makedirs(args.result_file_folder)



#load source trianing, target testing loader    
dataloader_source_train, dataloader_target_test = get_loader(args)
#1: pack test imgs to npz file
if os.path.exists(args.image_test_npz_file)==False:
    make_test_npz_file(dataloader_target_test,args)  
print ("loading test images")    
dataset = Test_Dataset(args.image_test_npz_file)
t_loader_test = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
#load cluster label loader
#1: pack imgs to npz file
if os.path.exists(args.image_npz_file)==False:
    make_npz_file(args)  
 
#2: initialize cluster label
if args.is_initialize == 1:
    print ("initialize cluster label")
    initialize_office_meta_learner(parser)
    #3: load initialize cluster label loader
    dataloader_cluster_label = get_cluster_loader(args)
else:
    #3: load provided cluster label loader
    dataloader_cluster_label = get_provided_cluster_loader(args) 


dataloader_no_shuffle = get_noshuffle_loader(args) 

if args.dataset_name == "OfficeHome":
    my_net = Alex_Model_OfficeHome().cuda()
else:
    my_net = Alex_Model_Office31().cuda()    

my_net.load_state_dict(torch.load("bvlc_model/bvlc_model.pth"),strict=False)    


loss_class = torch.nn.CrossEntropyLoss().cuda()
loss_domain = torch.nn.BCEWithLogitsLoss().cuda()


count = 0

my_net.train() 
for p in my_net.parameters():
    p.requires_grad = True
 

s_loader = iter(dataloader_source_train)
t_loader = iter(dataloader_cluster_label)


gamma =0.001
power = 0.75

try:
    with tqdm(total=args.max_iter, dynamic_ncols=True) as pbar: 
        for iter_count in range(args.max_iter):
            try:
                pbar.update(1)
            except IOError:
                pbar.update(1)  
            try:
                t_imgs, t_labels = t_loader.next()
            except StopIteration:
                t_loader = iter(dataloader_cluster_label)
                t_imgs, t_labels = t_loader.next()
            except OSError:
                t_loader = iter(dataloader_cluster_label)
                t_imgs, t_labels = t_loader.next() 
                
            try:
                s_imgs, s_labels = s_loader.next()
            except StopIteration:
                s_loader = iter(dataloader_source_train)
                s_imgs, s_labels = s_loader.next()  
            except OSError:
                s_loader = iter(dataloader_source_train)
                s_imgs, s_labels = s_loader.next()     
            s_imgs, s_labels = Variable(s_imgs.cuda()), Variable(s_labels.cuda()) 
            t_imgs, t_labels = Variable(t_imgs.cuda()), Variable(t_labels.cuda())
            t_labels = t_labels.long()
            
            p = min(1.0,float(iter_count*1.0) / (20000.0)) 
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            c_lr = args.lr*(1+gamma*iter_count)**(-power)
            
            #set optim
            optim_net = optim.SGD([
                    {'params': my_net.feature.conv1.weight , 'lr': 1.0*c_lr },
                    {'params': my_net.feature.conv1.bias   , 'lr': 2.0*c_lr },
                    {'params': my_net.feature.conv2.weight , 'lr': 1.0*c_lr  },
                    {'params': my_net.feature.conv2.bias   , 'lr': 2.0*c_lr  },
                    {'params': my_net.feature.conv3.weight , 'lr': 1.0*c_lr  },
                    {'params': my_net.feature.conv3.bias   , 'lr': 2.0*c_lr  },
                    {'params': my_net.feature.conv4.weight , 'lr': 1.0*c_lr  },
                    {'params': my_net.feature.conv4.bias   , 'lr': 2.0*c_lr  },
                    {'params': my_net.feature.conv5.weight , 'lr': 1.0*c_lr  },
                    {'params': my_net.feature.conv5.bias   , 'lr': 2.0*c_lr  },
                    {'params': my_net.feature.fc6.weight   , 'lr': 1.0*c_lr  },
                    {'params': my_net.feature.fc6.bias     , 'lr': 2.0*c_lr  },
                    {'params': my_net.feature.fc7.weight   , 'lr': 1.0*c_lr  },
                    {'params': my_net.feature.fc7.bias     , 'lr': 2.0*c_lr  },
                    {'params': my_net.bottleneck.weight             , 'lr': 10.0*c_lr  },
                    {'params': my_net.bottleneck.bias               , 'lr': 20.0*c_lr  },
                    {'params': my_net.class_classifier.fc8.weight , 'lr': 10.0*c_lr  },
                    {'params': my_net.class_classifier.fc8.bias   , 'lr': 20.0*c_lr  },
                    {'params': my_net.domain_classifier.dc_ip1.weight       , 'lr': 10.0*c_lr  },
                    {'params': my_net.domain_classifier.dc_ip1.bias         , 'lr': 20.0*c_lr  },
                    {'params': my_net.domain_classifier.dc_ip2.weight       , 'lr': 10.0*c_lr  },
                    {'params': my_net.domain_classifier.dc_ip2.bias         , 'lr': 20.0*c_lr  },
                    {'params': my_net.dc_ip3_st.weight       , 'lr': 10.0*c_lr  },
                    {'params': my_net.dc_ip3_st.bias         , 'lr': 20.0*c_lr  },
                    {'params': my_net.dc_ip3_mt.weight       , 'lr': 10.0*c_lr  },
                    {'params': my_net.dc_ip3_mt.bias         , 'lr': 20.0*c_lr  }
                ], lr=c_lr,momentum= 0.9)
            
        
            
            my_net.zero_grad()
            # training model using source data
            batch_size = len(s_labels)
            domain_s_label = torch.zeros(batch_size)
            domain_s_label = domain_s_label.float()
            domain_s_label = Variable(domain_s_label.cuda())
            
            class_output, _ , _,_ = my_net(input_data=s_imgs, alpha=alpha)
            
            #calculate Vst
            #calculate Classification loss
            Classification_loss = loss_class(class_output, s_labels)
            # training model using target data
            batch_size = len(t_imgs)
            domain_t_label = torch.ones(batch_size)
            domain_t_label = domain_t_label.float()
            domain_t_label = Variable(domain_t_label.cuda())  
            
            
            imgs = torch.cat((s_imgs, t_imgs,), 0)
            
            labels = torch.cat((domain_s_label,domain_t_label,), 0)
            
            
            _, domain_output, _,_ = my_net(input_data=imgs, alpha=alpha) 
            #calculate Adversarial DA loss
            labels = torch.unsqueeze(labels, 1)
            Adversarial_DA_loss = loss_domain(domain_output, labels)
            #calculate Vst
            Vst = Classification_loss + args.lambda_val * Adversarial_DA_loss
        
            t_class_output,_, domain_output_mt,_ = my_net(t_imgs,alpha=alpha)
            
            #calculate Vmt
            if args.dataset_name == "OfficeHome":
                Vmt = loss_class(domain_output_mt, t_labels)
            else:
                t_labels = torch.unsqueeze(t_labels, 1)
                t_labels = t_labels.float()
                Vmt = loss_domain(domain_output_mt, t_labels)
            #calculate Lent
            Lent = loss_entropy(t_class_output)
            #calculate Lvir
            vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)
            s_vat_loss = vat_loss(my_net,s_imgs)
            t_vat_loss = vat_loss(my_net,t_imgs)
            Lvir = s_vat_loss + args.rho_val * t_vat_loss
                    
            loss = Vst + args.gamma_val*Vmt + args.beta_val*Lent + Lvir 
            loss.backward()
            optim_net.step()              
            
            
            if iter_count%10 == 0:
                tqdm.write("iter is : {}  Classification loss is :{:.3f} Lent is :{:.3f}  Adversarial DA loss is :{:.3f} Vmt is:{:.3f} gamma_val is:{:.3f}".format(iter_count,Classification_loss.item(),\
                    Lent.item(), Adversarial_DA_loss.item(),Vmt.item(), args.gamma_val))
                print_log(iter_count+1, \
                Classification_loss.item(), \
                Lent.item(), \
                Adversarial_DA_loss.item(),\
                0,\
                0,\
                Vmt.item(),\
                0,\
                args.tmp_accuracy,\
                args.gamma_val,\
                ploter, iter_count+1)
                    
            # Meta-update     
            if (iter_count%args.update_meta_iter == 0)and(iter_count>1):
                torch.save(my_net.state_dict(), args.snapshot_model_name)
                dataloader_cluster_label = update_teacher(dataloader_no_shuffle,parser)
                t_loader = iter(dataloader_cluster_label)
                
            
            if (iter_count%args.test_iter == 0)and(iter_count>1):        
                torch.save(my_net.state_dict(), args.snapshot_model_name)
                args.tmp_accuracy = test_model(t_loader_test,iter_count,args)
                
                #save max accuracy model
                if args.tmp_accuracy > args.max_accuracy:
                    args.max_accuracy = args.tmp_accuracy 
                    args.max_accuracy_iter = iter_count
                    torch.save(my_net.state_dict(), args.snapshot_max_accuracy_model_name)
                    result = open(args.result_file_name,"a")
                    result.write("max accuracy is :"+str("%.5f" % (args.max_accuracy*100.0))+\
                        "   iter is :"+str(iter_count)+"\n")
                    result.close()      
except OSError:
    print "OSError"
except KeyboardInterrupt:
    pbar.close()
    raise
pbar.close()          
print ("max accuracy is : ",str(args.max_accuracy))          
print ("max equal weight accuracy is : ",str(test_model_equal_weight(args)))            
shutil.copy(args.snapshot_max_accuracy_model_name,\
    args.snapshot_model_file+args.task_name+'_accuracy_is_'+str("%.3f" %  (args.max_accuracy*100.0))+'.pth')    

sys.exit()



