import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F


@contextlib.contextmanager
def _disable_tracking_bn_stats(model):

    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
            
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)


def _l2_normalize(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d


def _kl_div(log_probs, probs):
    # pytorch KLDLoss is averaged over all dim if size_average=True
    kld = F.kl_div(log_probs, probs, size_average=False)
    return kld / log_probs.shape[0]

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
    
class VATLoss(nn.Module):

    def __init__(self, xi=10.0, eps=1.0, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, model, x):
        with torch.no_grad():  
            tmp,_,_,_= model(x,0)
            #pred = F.softmax(tmp, dim=1)
            pred = tmp

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(
            torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        d = _l2_normalize(d)

        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat,_ ,_,_= model(x + self.xi * d,0)
                #adv_distance = _kl_div(F.log_softmax(pred_hat, dim=1), pred)
                adv_distance = two_loss_entropy(pred_hat, pred)
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                model.zero_grad()
    
            # calc LDS
            r_adv = d * self.eps
            #print r_adv           
            pred_hat ,_,_,_ = model(x + r_adv,0)
            #lds = _kl_div(F.log_softmax(pred_hat, dim=1), pred)
            lds = two_loss_entropy(pred_hat, pred)
            

        return lds