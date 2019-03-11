# -*- coding: utf-8 -*-
#
# Copyright © dawnranger.
#
# 2018-05-08 10:15 <dawnranger123@gmail.com>
#
# Distributed under terms of the MIT license.
from __future__ import division, print_function
import numpy as np
import tqdm
import torch
from torch.utils.data import Dataset


class Image_Dataset(Dataset):

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
    f.close()
    x = x_train
    y = y_train.astype(np.int32)
    x = x.reshape((x.shape[0], -1)).astype(np.float32)
    x = np.divide(x, 255.)
    print('samples', x.shape)
    return x, y              
            
            

def count_difference(idec_args,y_pred):
    if idec_args.dataset_name == "Office31":
        list = [0,0]
        for num in y_pred:
            list[num]+=1
    elif  idec_args.dataset_name == "OfficeHome":
        list = [0,0,0]
        for num in y_pred:
            list[num]+=1
    else:
        list = [0,0,0,0]
        for num in y_pred:
            list[num]+=1
    #return abs(list[0]-list[1])
    return np.std(list)  



def write_list(y_pred,idec_args):
    target_name_dict = {
        'mnist': 'mm_svhn_synth_usps', 
        'mnist_m': 'mnist_svhn_usps_synth', 
        'svhn': 'mnist_mm_usps_synth', 
        'synth': 'mnist_svhn_usps_mm', 
        'usps': 'mnist_svhn_synth_mm',
        'A': 'RPC', 
        'C': 'RPA', 
        'P': 'RCA', 
        'R': 'PCA', 
        'amazon': 'DW', 
        'webcam': 'AD', 
        'dslr': 'AW',         
    } 
    if idec_args.dataset_name == "OfficeHome":
        fin = open("dataset/OfficeHome/"+target_name_dict[idec_args.source_name]+"_domain_List.txt", "r") 
    elif idec_args.dataset_name == "Office31":
        fin = open("dataset/Office31/"+target_name_dict[idec_args.source_name]+"_domain_list_equal.txt", "r") 
    else:
        fin = open("dataset/image_list/"+target_name_dict[idec_args.source_name]+"_train.txt", "r")     
    fout = open(idec_args.update_list_file+"/cluster_label.txt","w")
    count = 0
    for line in fin:
        data = line.strip().split(" ")
        fout.write(data[0]+" "+str(y_pred[count])+"\n")
        count +=1
        
    fin.close()
    fout.close()
#######################################################
# Evaluate Critiron
#######################################################


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
