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
from dataset.data_loader import get_loader,get_cluster_loader,get_provided_cluster_loader
from torchvision import datasets
from torchvision import transforms
from models.model import Model,Domain_Classifier
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from vat import VATLoss
from utils import LinePlotter
from function import *
import shutil
import time
time = str(time.strftime('%Y_%m_%d %H:%M:%S',time.localtime(time.time())))
import argparse
sys.path.append("../IDEC")
from idec import *

#2: initialize hyper-parameter and environment
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", help="set dataset name", default="digit_five")
parser.add_argument("--source_name", help="set source domain, form mnist, mnist_m, svhn, synth, usps", default="mnist")
parser.add_argument("--batch_size", help="set batch size", type=int, default=100)
parser.add_argument("--update_meta_iter", help="set update meta iter number", type=int, default=10000)
parser.add_argument("--test_iter", help="set test iter number", type=int, default=100)
parser.add_argument("--snapshot", help="set which folder to save model and result", default="snapshot/")
parser.add_argument("--lr", help="set learning rate", type=int, default=1e-3)
parser.add_argument("--gpu_id", help="set gpu id", type=int, default=0)
parser.add_argument("--is_in", help="whether use ", type=int, default=1)
parser.add_argument("--image_size", help="set input image size", type=int, default=32)
parser.add_argument("--max_iter", help="set max iter times", type=int, default = 200000)
parser.add_argument("--beta_val", help="set beta value", type=int, default= 1e-2 )
parser.add_argument("--lambda_val", help="set lambda value", type=int, default= 1e-2)
parser.add_argument("--gamma_val", help="set gamma value", type=int, default= 0.0)
parser.add_argument("--rho_val", help="set rho value", type=int, default= 1e-2)
parser.add_argument("--is_initialize", help="whether initialize cluster labels or use provided cluster labels", type=int,default= 0)
parser.add_argument("--data_root", help="set dataset file", default=('dataset'))
parser.add_argument("--task_name", help="generate this task's name", \
    default= parser.parse_args().source_name+'_2_all')
parser.add_argument("--snapshot_file", help="generate folder to save models and result", \
    default='snapshot/'+parser.parse_args().source_name+'_2_other_'+time )
parser.add_argument("--update_list_file", \
    default= 'snapshot/'+parser.parse_args().source_name+'_2_other_'+time+'/update_list/' )
parser.add_argument("--snapshot_model_file", \
    default= 'snapshot/'+parser.parse_args().source_name+'_2_other_'+time+'/model/' )
parser.add_argument("--snapshot_model_name", \
    default= parser.parse_args().snapshot_model_file+parser.parse_args().task_name+'_current_model.pth')
parser.add_argument("--snapshot_max_accuracy_model_name", \
    default= parser.parse_args().snapshot_model_file+parser.parse_args().task_name+'_max_accuracy_model.pth')
parser.add_argument("--result_file_folder",  \
    default= 'snapshot/'+parser.parse_args().source_name+'_2_other_'+time+'/result/')
parser.add_argument("--result_file_name", \
    default= 'snapshot/'+parser.parse_args().source_name+'_2_other_'+time+'/result/result.txt')
parser.add_argument("--image_npz_file", help="pack images to npz flie", \
    default= 'dataset/image_npz/'+parser.parse_args().source_name+'.npz' )
parser.add_argument("--image_npz_update_file", \
    default= 'dataset/image_npz/'+parser.parse_args().source_name+'_update.npz' )
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




#load source, target(trian/test) loader    
dataloader_source_train,dataloader_source_test,\
    dataloader_target_train,dataloader_target_train_noshuffle,\
    dataloader_target_test = get_loader(args)   

#load cluster label loader
#1: pack imgs to npz file
if os.path.exists(args.image_npz_file)==False:
    make_npz_file(args)
#2: initialize cluster label
if args.is_initialize == 1:
    print ("initialize cluster label")
    initialize_digit_meta_learner(parser)
    #3: load initialize cluster label loader
    dataloader_cluster_label = get_cluster_loader(args)
else:
    #3: load provided cluster label loader
    dataloader_cluster_label = get_provided_cluster_loader(args)    




   
    
# initialize model and domain_classifier
model = Model(args.is_in,args.batch_size).cuda()
domain_classifier = Domain_Classifier().cuda()

# initialize data loader iter
s_loader = iter(dataloader_source_train)
t_loader = iter(dataloader_target_train)
c_loader = iter(dataloader_cluster_label)






try:
    with tqdm(total=args.max_iter, dynamic_ncols=True) as pbar: 
        for iter_count in range(args.max_iter):
            try:
                pbar.update(1)
            except IOError:
                pbar.update(1)    
            #get training batch
            try:
                s_imgs, s_labels = s_loader.next()
            except StopIteration:
                s_loader = iter(dataloader_source_train)
                s_imgs, s_labels = s_loader.next()  
            except OSError:
                s_loader = iter(dataloader_source_train)
                s_imgs, s_labels = s_loader.next()
            try:
                t_imgs, t_labels = t_loader.next()
            except StopIteration:
                t_loader = iter(dataloader_target_train)
                t_imgs, t_labels = t_loader.next()
            except OSError:
                t_loader = iter(dataloader_target_train)
                t_imgs, t_labels = t_loader.next()    
            try:
                cluster_imgs, cluster_labels = c_loader.next()
            except StopIteration:
                c_loader = iter(dataloader_cluster_label)
                cluster_imgs, cluster_labels = c_loader.next()
            except OSError:
                c_loader = iter(dataloader_cluster_label)
                cluster_imgs, cluster_labels = c_loader.next()    
                
            s_imgs, s_labels = Variable(s_imgs.cuda()), Variable(s_labels.cuda())
            t_imgs, t_labels = Variable(t_imgs.cuda()), Variable(t_labels.cuda())
            cluster_imgs, cluster_labels = Variable(cluster_imgs.cuda()), Variable(cluster_labels.cuda())
            
            model.train()
            
            #update F and C
            optim_m = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
            optim_m.zero_grad()
            s_feature,s_label_pred = model(s_imgs)
            Classification_loss = loss_class(s_label_pred, s_labels)
            
            t_feature,t_label_pred = model(t_imgs)
            #calculate Lent
            Lent = loss_entropy(t_label_pred)
            
            #d_1: determine image from source or target    
            #d_2: determine target image from which sub-target
            s_feature_pred = s_label_pred
            t_feature_pred = t_label_pred
            
            s_d_1,s_d_2 = domain_classifier(s_feature,s_feature_pred)
            t_d_1,t_d_2 = domain_classifier(t_feature,t_feature_pred)
            
            real_logit = torch.squeeze(s_d_1)
            fake_logit = torch.squeeze(t_d_1)
            
            #calculate V~mt  
            V_tilde_mt = -1.0*loss_entropy(t_d_2)
            
            domain_label_real = Variable(torch.zeros(s_imgs.size()[0]).float().cuda())
            domain_label_fake = Variable(torch.ones(t_imgs.size()[0]).float().cuda())
            Adversarial_DA_loss = 0.5*(loss_domain(real_logit,domain_label_real)+loss_domain(fake_logit,domain_label_fake))  
            #calculate Vst
            Vst = Classification_loss + args.lambda_val*Adversarial_DA_loss
            
            #calculate Lvir
            vat_loss = VATLoss(xi=10.0, eps=1.0, ip=1)
            s_val_loss = vat_loss(model,s_imgs)
            t_val_loss = vat_loss(model,t_imgs) 
            Lvir = s_val_loss + args.rho_val*t_val_loss
            
            #update F and C
            args.gamma_val = (iter_count*1.0)/args.max_iter
            loss = Vst + args.gamma_val * V_tilde_mt + args.beta_val * Lent + Lvir       
            loss.backward(retain_graph=True)
            optim_m.step()
            
            
            
            #update Dst and Dmt
            optim_d = optim.Adam(domain_classifier.parameters(), lr=args.lr, betas=(0.5, 0.999))
            domain_classifier.train()
            domain_classifier.zero_grad()
            
            #d_1: determine image from source or target    
            #d_2: determine target image from which sub-target
            s_d_1,s_d_2 = domain_classifier(s_feature,s_feature_pred)
            t_d_1,t_d_2 = domain_classifier(t_feature,t_feature_pred)
            
            real_logit = torch.squeeze(s_d_1)    
            fake_logit = torch.squeeze(t_d_1)
            domain_label_real = Variable(torch.ones(s_imgs.size()[0]).float().cuda())
            domain_label_fake = Variable(torch.zeros(t_imgs.size()[0]).float().cuda())
            
            # target domain class label pred 
            d_label_feature,d_label_pred = model(cluster_imgs)
            d_feature_pred = d_label_pred
            _,d_label_d_2 = domain_classifier(d_label_feature,d_feature_pred)
            
            
            # calculate confusion loss
            Lcf = loss_log(s_d_2)
            
            # calculate Vmt
            Vmt = loss_class(d_label_d_2, cluster_labels)
            
            # calculate Vst
            Adversarial_DA_loss_Dst = 0.5*(loss_domain(real_logit,domain_label_real)+ \
                loss_domain(fake_logit,domain_label_fake))
            D_loss = Vmt+Lcf
            
            #update Dst and Dmt
            Adversarial_DA_loss_Dst.backward(retain_graph=True)  
            D_loss.backward()     
            optim_d.step()
            
        
            if iter_count%10 == 0:
                print_log(iter_count+1, \
                Classification_loss.item(), \
                Lent.item(), \
                Adversarial_DA_loss.item(),\
                Adversarial_DA_loss_Dst.item(),\
                Lcf.item(),\
                Vmt.item(),\
                V_tilde_mt.item(),\
                args.tmp_accuracy,\
                args.gamma_val,\
                ploter, iter_count+1)
            
            # Meta-update    
            if (iter_count % args.update_meta_iter == 0)and(iter_count>1):
                torch.save(model.state_dict(), args.snapshot_model_name)
                dataloader_cluster_label = update_teacher(dataloader_target_train_noshuffle,parser)
                c_loader = iter(dataloader_cluster_label)
            
            # test model accuracy on target test data    
            if iter_count % args.test_iter== 0:
                print("iter is : {}  Classification loss is :{:.3f} Lent is :{:.3f}  Adversarial DA loss is :{:.3f} Adversarial_DA_loss_Dst is:{:.3f}  confusion loss is:{:.3f} Vmt is:{:.3f}  V_tilde_mt is:{:.3f} gamma_val is:{:.3f}".format(iter_count,Classification_loss.item(),\
                    Lent.item(), Adversarial_DA_loss.item(), Adversarial_DA_loss_Dst.item(), Lcf.item(),\
                    Vmt.item(), V_tilde_mt.item(), args.gamma_val))    
                torch.save(model.state_dict(), args.snapshot_model_name)
                args.tmp_accuracy = test_model(dataloader_target_test,iter_count,args)
                
                #save max accuracy model
                if args.tmp_accuracy > args.max_accuracy:
                    args.max_accuracy = args.tmp_accuracy 
                    args.max_accuracy_iter = iter_count
                    torch.save(model.state_dict(), args.snapshot_max_accuracy_model_name)
                    result = open(args.result_file_name,"a")
                    result.write("max accuracy is :"+str("%.5f" % (args.max_accuracy*100.0))+\
                        "   iter is :"+str(iter_count)+"\n")
                    result.close()
except KeyboardInterrupt:
    pbar.close()
    raise
pbar.close()
    
print ("max accuracy is : ",str(args.max_accuracy))          
print ("max equal weight accuracy is : ",str(test_model_equal_weight(args)))            
shutil.copy(args.snapshot_max_accuracy_model_name,\
    args.snapshot_model_file+args.task_name+'_accuracy_is_'+str("%.3f" %  (args.max_accuracy*100.0))+'.pth')    

sys.exit()




    
    
    


