import torch
import torch.nn.functional as F

def get_dis_loss(dis_fake, dis_real):
    D_loss = torch.mean(dis_fake ** 2) + torch.mean((dis_real - 1) ** 2)
    return D_loss

def get_confusion_loss(dis_common):
    confusion_loss = torch.mean((dis_common - 0.5) ** 2)
    return confusion_loss

def get_cls_loss(pred, gt):
    cls_loss = F.nll_loss(pred, gt)
    return cls_loss

