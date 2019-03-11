# -*- coding: utf-8 -*-
#
# Copyright © dawnranger.
#
# 2018-05-08 10:15 <dawnranger123@gmail.com>
#
# Distributed under terms of the MIT license.
from __future__ import print_function, division
import argparse
import numpy as np
from sklearn.cluster import KMeans
import random
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
import os
import sys
from idec_utils import Image_Dataset,cluster_acc,count_difference,write_list
from torchvision import transforms


class AE(nn.Module):   
    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                n_input, n_z):
        super(AE, self).__init__()
    
        # encoder
        self.enc_1 = nn.Linear(n_input, n_enc_1)
        self.enc_2 = nn.Linear(n_enc_1, n_enc_2)
        self.enc_3 = nn.Linear(n_enc_2, n_enc_3)
    
        self.z_layer = nn.Linear(n_enc_3, n_z)
    
        # decoder
        self.dec_1 = nn.Linear(n_z, n_dec_1)
        self.dec_2 = nn.Linear(n_dec_1, n_dec_2)
        self.dec_3 = nn.Linear(n_dec_2, n_dec_3)
    
        self.x_bar_layer = nn.Linear(n_dec_3, n_input)
    
    def forward(self, x):
    
        # encoder
        enc_h1 = F.relu(self.enc_1(x))
        enc_h2 = F.relu(self.enc_2(enc_h1))
        enc_h3 = F.relu(self.enc_3(enc_h2))
    
        z = self.z_layer(enc_h3)
    
        # decoder
        dec_h1 = F.relu(self.dec_1(z))
        dec_h2 = F.relu(self.dec_2(dec_h1))
        dec_h3 = F.relu(self.dec_3(dec_h2))
        x_bar = self.x_bar_layer(dec_h3)
    
        return x_bar, z


class IDEC(nn.Module):

    def __init__(self,
                n_enc_1,
                n_enc_2,
                n_enc_3,
                n_dec_1,
                n_dec_2,
                n_dec_3,
                n_input,
                n_z,
                n_clusters,
                alpha=1,
                pretrain_path='data/ae_mnist.pkl'):
        super(IDEC, self).__init__()
        self.alpha = 1.0
        self.pretrain_path = pretrain_path

        self.ae = AE(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self,dataset,idec_args):
        pretrain_ae(self.ae,dataset,idec_args)
        # load pretrain weights
        self.ae.load_state_dict(torch.load(self.pretrain_path))
        print('load pretrained ae from', self.pretrain_path)

    def forward(self, x):

        x_bar, z = self.ae(x)
        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x_bar, q


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def pretrain_ae(model,dataset,idec_args):
    '''
    pretrain autoencoder
    '''
    train_loader = DataLoader(dataset, batch_size=idec_args.idec_batch_size, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=idec_args.idec_lr)
    print ("pretrain process")
    for epoch in tqdm(range(idec_args.pretrain_epoch)):
        total_loss = 0.
        for batch_idx, (x, _,_) in tqdm(enumerate(train_loader)):
            x = x.cuda()
            x = x.view(x.size()[0],-1)
            #print (x.size())
            
            optimizer.zero_grad()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
        
        tqdm.write("epoch {} loss={:.4f}".format(epoch,
                                            total_loss / (batch_idx + 1)))                                   
        torch.save(model.state_dict(), idec_args.pretrain_path)
    print("model saved to {}.".format(idec_args.pretrain_path))

def train_idec(idec_args,dataset):
    
    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    model = IDEC(
        n_enc_1=500,
        n_enc_2=500,
        n_enc_3=1000,
        n_dec_1=1000,
        n_dec_2=500,
        n_dec_3=500,
        
        
        n_input=idec_args.n_input,
        n_z=idec_args.n_z,
        n_clusters=idec_args.n_clusters,
        alpha=1.0,
        pretrain_path=idec_args.pretrain_path).cuda()

    model.pretrain(dataset,idec_args)
    train_loader = DataLoader(
        dataset, batch_size=idec_args.idec_batch_size, shuffle=False)   
    optimizer = Adam(model.parameters(), lr=idec_args.idec_lr)

    # cluster parameter initiate
    data = dataset.x
    y = dataset.y
    
    
    for batch_idx, (x, _, _) in enumerate(train_loader):
        x = x.cuda()
        _, tmp_hidden = model(x)
        if batch_idx==0:
            hidden = tmp_hidden.data   
        else:
            hidden = torch.cat((hidden,tmp_hidden.data), 0)
    kmeans = KMeans(n_clusters=idec_args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())    
    nmi_k = nmi_score(y_pred, y)
    print("nmi score={:.4f}".format(nmi_k))

    hidden = None
    x_bar = None

    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).cuda()

    model.train()
    
    print ("training process")
    for epoch in tqdm(range(idec_args.train_epoch)):
            
        for batch_idx, (x, _, _) in enumerate(train_loader):
            x = x.cuda()
            _, tmp_q = model(x)
            
            # update target distribution p
            tmp_q = tmp_q.data
                
            if batch_idx==0:
                concat_q = tmp_q
            else:
                concat_q = torch.cat((concat_q,tmp_q), 0)
        p = target_distribution(concat_q)          
        
        idec_args.eval = 0
        if idec_args.eval == 1:
            # evaluate clustering performance            
            y_pred = concat_q.cpu().numpy().argmax(1)
            delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]
            y_pred_last = y_pred      
            
            acc = cluster_acc(y, y_pred)
            if acc>idec_args.max_acc:
                idec_args.max_acc = acc
                idec_args.max_acc_iter = epoch
            tqdm.write("acc is : {:.3f}".format(acc))
        
        
        difference = count_difference(idec_args,y_pred)
        tqdm.write("difference is : {:.3f}".format(difference))
        if difference < idec_args.min_difference:
            idec_args.min_difference = difference
            idec_args.min_difference_iter = epoch
            final_y_pred = y_pred
            # generate cluster label txt file
            write_list(final_y_pred,idec_args) 
        
        for batch_idx, (x, _, idx) in enumerate(train_loader):

            x = x.cuda()
            idx = idx.cuda()

            x_bar, q = model(x)

            reconstr_loss = F.mse_loss(x_bar, x)
            kl_loss = F.kl_div(q.log(), p[idx])
            loss = idec_args.gamma * kl_loss + reconstr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    if idec_args.dataset_name == "digit_five":
        final_y_pred = y_pred
        # generate cluster label txt file
        write_list(final_y_pred,idec_args)


def initialize_digit_meta_learner(parser):
    idec_parser = argparse.ArgumentParser(
        add_help=False,
        description='train',
        parents= [parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    idec_parser.add_argument('--idec_lr', type=float, default=0.001)
    idec_parser.add_argument('--idec_batch_size', default=256, type=int)
    idec_parser.add_argument(
        '--gamma',
        default=0.1,
        type=float,
        help='coefficient of clustering loss')
    idec_parser.add_argument('--update_interval', default=1, type=int)
    idec_parser.add_argument('--eval', default=0, type=int)
    idec_parser.add_argument('--min_difference', default=1e+6, type=int)
    idec_parser.add_argument('--min_difference_acc', default=0, type=int)
    idec_parser.add_argument('--max_acc', default=0, type=int)
    idec_parser.add_argument('--min_difference_iter', default=0, type=int)
    idec_parser.add_argument('--max_acc_iter', default=0, type=int)
    idec_parser.add_argument('--tol', default=0.001, type=float)
    idec_args = idec_parser.parse_args()
    
    
    idec_args.cuda = True
    idec_args.pretrain_path = '../IDEC/data/'+idec_args.model_name+idec_args.source_name+'.pkl'
    if (idec_args.source_name == "usps") or (idec_args.source_name == "synth"):
        idec_args.pretrain_epoch = 2
    else:
        idec_args.pretrain_epoch = 50
    idec_args.train_epoch = 50
    idec_args.n_clusters = 4
    idec_args.n_z = 4
    idec_args.n_input = 2352
    dataset = Image_Dataset(idec_args.image_npz_file) 
        
    train_idec(idec_args,dataset)

        
def update_digit_meta_learner(parser):
    idec_parser = argparse.ArgumentParser(
        add_help=False,
        description='train',
        parents= [parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    idec_parser.add_argument('--idec_lr', type=float, default=0.001)
    idec_parser.add_argument('--idec_batch_size', default=256, type=int)
    idec_parser.add_argument(
        '--gamma',
        default=0.1,
        type=float,
        help='coefficient of clustering loss')
    idec_parser.add_argument('--update_interval', default=1, type=int)
    idec_parser.add_argument('--eval', default=0, type=int)
    idec_parser.add_argument('--min_difference', default=1e+6, type=int)
    idec_parser.add_argument('--min_difference_acc', default=0, type=int)
    idec_parser.add_argument('--max_acc', default=0, type=int)
    idec_parser.add_argument('--min_difference_iter', default=0, type=int)
    idec_parser.add_argument('--max_acc_iter', default=0, type=int)
    idec_parser.add_argument('--tol', default=0.001, type=float)
    idec_args = idec_parser.parse_args()
    
    
    idec_args.cuda = True
    idec_args.pretrain_path = '../IDEC/data/'+idec_args.model_name+idec_args.source_name+'.pkl'
    idec_args.pretrain_epoch = 50
    idec_args.train_epoch = 50
    idec_args.n_clusters = 4
    idec_args.n_z = 4
    idec_args.n_input = (64*8*8+10+28*28*3)
    dataset = Image_Dataset(idec_args.image_npz_update_file) 
        
    train_idec(idec_args,dataset)

    
def initialize_office_meta_learner(parser):
    idec_parser = argparse.ArgumentParser(
        add_help=False,
        description='train',
        parents= [parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    idec_parser.add_argument('--idec_lr', type=float, default=0.001)
    idec_parser.add_argument('--idec_batch_size', default=256, type=int)
    idec_parser.add_argument(
        '--gamma',
        default=0.1,
        type=float,
        help='coefficient of clustering loss')
    idec_parser.add_argument('--update_interval', default=1, type=int)
    idec_parser.add_argument('--eval', default=0, type=int)
    idec_parser.add_argument('--min_difference', default=1e+6, type=int)
    idec_parser.add_argument('--min_difference_acc', default=0, type=int)
    idec_parser.add_argument('--max_acc', default=0, type=int)
    idec_parser.add_argument('--min_difference_iter', default=0, type=int)
    idec_parser.add_argument('--max_acc_iter', default=0, type=int)
    idec_parser.add_argument('--tol', default=0.001, type=float)
    idec_args = idec_parser.parse_args()
    
    
    idec_args.cuda = True
    idec_args.pretrain_path = '../IDEC/data/'+idec_args.model_name+idec_args.source_name+'.pkl'
    
    
    if idec_args.dataset_name == "OfficeHome":
        idec_args.pretrain_epoch = 100
        idec_args.train_epoch = 50
        idec_args.n_clusters = 3
        idec_args.n_z = 3
        idec_args.n_input = 12288
        dataset = Image_Dataset(idec_args.image_npz_file)
        train_idec(idec_args,dataset)        
    else:
        idec_args.pretrain_epoch = 2
        idec_args.train_epoch = 20
        idec_args.n_clusters = 2
        idec_args.n_z = 2
        idec_args.n_input = 3*64*64
        dataset = Image_Dataset(idec_args.image_npz_file)  
        for i in tqdm(range(20)):    
            train_idec(idec_args,dataset)
        print ("min_difference is :",str(idec_args.min_difference))
        print ("min_difference_acc is :",str(idec_args.min_difference_acc))
        sys.exit()
    
def update_office_meta_learner(parser):
    idec_parser = argparse.ArgumentParser(
        add_help=False,
        description='train',
        parents= [parser],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    idec_parser.add_argument('--idec_lr', type=float, default=0.001)
    idec_parser.add_argument('--idec_batch_size', default=256, type=int)
    idec_parser.add_argument(
        '--gamma',
        default=0.1,
        type=float,
        help='coefficient of clustering loss')
    idec_parser.add_argument('--update_interval', default=1, type=int)
    idec_parser.add_argument('--eval', default=0, type=int)
    idec_parser.add_argument('--min_difference', default=1e+6, type=int)
    idec_parser.add_argument('--min_difference_acc', default=0, type=int)
    idec_parser.add_argument('--max_acc', default=0, type=int)
    idec_parser.add_argument('--min_difference_iter', default=0, type=int)
    idec_parser.add_argument('--max_acc_iter', default=0, type=int)
    idec_parser.add_argument('--tol', default=0.001, type=float)
    idec_args = idec_parser.parse_args()
    
    
    idec_args.cuda = True
    idec_args.pretrain_path = '../IDEC/data/'+idec_args.model_name+idec_args.source_name+'.pkl'
    
    if idec_args.model_name == "resnet":
        if idec_args.dataset_name == "OfficeHome":
            idec_args.pretrain_epoch = 50
            idec_args.train_epoch = 20
            idec_args.n_clusters = 3
            idec_args.n_z = 3
            idec_args.n_input = 4096*2+65+64*64*3
            dataset = Image_Dataset(idec_args.image_npz_update_file)     
        else:
            idec_args.pretrain_epoch = 50
            idec_args.train_epoch = 20
            idec_args.n_clusters = 2
            idec_args.n_z = 2
            idec_args.n_input = 4096*2+31+64*64*3
            dataset = Image_Dataset(idec_args.image_npz_update_file)  
    else:        
        if idec_args.dataset_name == "OfficeHome":
            idec_args.pretrain_epoch = 50
            idec_args.train_epoch = 20
            idec_args.n_clusters = 3
            idec_args.n_z = 3
            idec_args.n_input = 4096+65+64*64*3
            dataset = Image_Dataset(idec_args.image_npz_update_file)     
        else:
            idec_args.pretrain_epoch = 50
            idec_args.train_epoch = 20
            idec_args.n_clusters = 2
            idec_args.n_z = 2
            idec_args.n_input = 4096+31+64*64*3
            dataset = Image_Dataset(idec_args.image_npz_update_file)     
            
    train_idec(idec_args,dataset)    