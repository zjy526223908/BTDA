import torch.nn as nn
import torch
from functions import ReverseLayerF

#print torch.normal(mean=torch.zeros(3, 4), std=torch.ones(3, 4), out=None)


tensor = torch.zeros(1, 10)
torch.nn.init.normal_(tensor, mean=0, std=0.1)

print tensor