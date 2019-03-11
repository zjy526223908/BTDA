import torch.nn as nn
from torchvision import models
import torch
from functions import ReverseLayerF
class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta

    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x

class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)
        
       
        
class Res_Model_OfficeHome(nn.Module):

    def __init__(self):
        super(Res_Model_OfficeHome, self).__init__()
        model_resnet50 = models.resnet50(pretrained=True)
        self.feature = nn.Sequential()
        #self.feature.add_module('instance_norm_1', nn.InstanceNorm2d((30,3,227,227)))
        self.feature.add_module('conv1', model_resnet50.conv1    )
        self.feature.add_module('bn1', model_resnet50.bn1        )
        self.feature.add_module('relu', model_resnet50.relu      )
        self.feature.add_module('maxpool', model_resnet50.maxpool)
        self.feature.add_module('layer1', model_resnet50.layer1  )
        self.feature.add_module('layer2', model_resnet50.layer2  )
        self.feature.add_module('layer3', model_resnet50.layer3  )
        self.feature.add_module('layer4', model_resnet50.layer4  )
        self.feature.add_module('avgpool', model_resnet50.avgpool)
        self.feature.add_module('view',View(-1, 2048 * 2 * 2))
        
        
        self.bottleneck = nn.Linear(2048 * 2 * 2, 256)
        
        
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('fc_65',nn.Linear(256, 65))
            

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('dc_ip1', nn.Linear(256 , 1024))  
        self.domain_classifier.add_module('dc_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('dc_drop1', nn.Dropout())
        self.domain_classifier.add_module('dc_ip2', nn.Linear(1024, 1024))  
        self.domain_classifier.add_module('dc_relu2', nn.ReLU(True))
        self.domain_classifier.add_module('dc_drop2', nn.Dropout())
        
        self.dc_ip3_d1 = nn.Linear(1024, 1)
        self.dc_ip3_d2 = nn.Linear(1024, 3)
        

    def forward(self, input_data, alpha):
        feature_0 = self.feature(input_data)
        feature = self.bottleneck(feature_0)
        

        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        
        
        domain_output_st = self.dc_ip3_d1(self.domain_classifier(reverse_feature))
        
        domain_output_mt = self.dc_ip3_d2(self.domain_classifier(reverse_feature))
        
        return class_output, domain_output_st,domain_output_mt,feature_0
        
        
class Res_Model_Office31(nn.Module):

    def __init__(self):
        super(Res_Model_Office31, self).__init__()
        model_resnet50 = models.resnet50(pretrained=True)
        self.feature = nn.Sequential()
        #self.feature.add_module('instance_norm_1', nn.InstanceNorm2d((30,3,227,227)))
        self.feature.add_module('conv1', model_resnet50.conv1    )
        self.feature.add_module('bn1', model_resnet50.bn1        )
        self.feature.add_module('relu', model_resnet50.relu      )
        self.feature.add_module('maxpool', model_resnet50.maxpool)
        self.feature.add_module('layer1', model_resnet50.layer1  )
        self.feature.add_module('layer2', model_resnet50.layer2  )
        self.feature.add_module('layer3', model_resnet50.layer3  )
        self.feature.add_module('layer4', model_resnet50.layer4  )
        self.feature.add_module('avgpool', model_resnet50.avgpool)
        self.feature.add_module('view',View(-1, 2048 * 2 * 2))
        
        
        self.bottleneck = nn.Linear(2048 * 2 * 2, 256)
        
        
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('fc_31',nn.Linear(256, 31))
            

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('dc_ip1', nn.Linear(256 , 1024))  
        self.domain_classifier.add_module('dc_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('dc_drop1', nn.Dropout())
        self.domain_classifier.add_module('dc_ip2', nn.Linear(1024, 1024))  
        self.domain_classifier.add_module('dc_relu2', nn.ReLU(True))
        self.domain_classifier.add_module('dc_drop2', nn.Dropout())
        
        self.dc_ip3_d1 = nn.Linear(1024, 1)
        self.dc_ip3_d2 = nn.Linear(1024, 1)
        

    def forward(self, input_data, alpha):
        feature_0 = self.feature(input_data)
        feature = self.bottleneck(feature_0)
        

        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        
        
        domain_output_st = self.dc_ip3_d1(self.domain_classifier(reverse_feature))
        
        domain_output_mt = self.dc_ip3_d2(self.domain_classifier(reverse_feature))
        
        

        return class_output, domain_output_st,domain_output_mt,feature_0        
        
        
        
class Alex_Model_Office31(nn.Module):

    def __init__(self):
        super(Alex_Model_Office31, self).__init__()
        self.feature = nn.Sequential()
        #self.feature.add_module('instance_norm_1', nn.InstanceNorm2d((30,3,227,227)))
        self.feature.add_module('conv1', nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0))       
        self.feature.add_module('relu1', nn.ReLU(True))
        self.feature.add_module('norm1', LRN(local_size=5, alpha=0.0001, beta=0.75))
        self.feature.add_module('pool1', nn.MaxPool2d(kernel_size=3, stride=2)) 
        
        self.feature.add_module('conv2', nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2))       
        self.feature.add_module('relu2', nn.ReLU(True))
        self.feature.add_module('norm2', LRN(local_size=5, alpha=0.0001, beta=0.75))
        self.feature.add_module('pool2', nn.MaxPool2d(kernel_size=3, stride=2))  
        
        self.feature.add_module('conv3', nn.Conv2d(256, 384, kernel_size=3, padding=1))       
        self.feature.add_module('relu3', nn.ReLU(True))
        
        self.feature.add_module('conv4', nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2))       
        self.feature.add_module('relu4', nn.ReLU(True))
        
        self.feature.add_module('conv5', nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2))       
        self.feature.add_module('relu5', nn.ReLU(True))
        self.feature.add_module('pool5', nn.MaxPool2d(kernel_size=3, stride=2)) 

        self.feature.add_module('view',View(-1, 256 * 6 * 6)) 
        self.feature.add_module('fc6',nn.Linear(256 * 6 * 6, 4096))
        self.feature.add_module('relu6',nn.ReLU(inplace=True))
        self.feature.add_module('drop6', nn.Dropout())
        self.feature.add_module('fc7',nn.Linear(4096, 4096))
        self.feature.add_module('relu7',nn.ReLU(inplace=True))
        self.feature.add_module('drop7', nn.Dropout())
        
        
        self.bottleneck = nn.Linear(4096, 256)
        
        
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('fc8',nn.Linear(256, 31))
        
        

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('dc_ip1', nn.Linear(256 , 1024))  
        self.domain_classifier.add_module('dc_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('dc_drop1', nn.Dropout())
        self.domain_classifier.add_module('dc_ip2', nn.Linear(1024, 1024))  
        self.domain_classifier.add_module('dc_relu2', nn.ReLU(True))
        self.domain_classifier.add_module('dc_drop2', nn.Dropout())
        
        self.dc_ip3_st = nn.Linear(1024, 1)
        
        
        self.dc_ip3_mt = nn.Linear(1024, 1)


    def forward(self, input_data, alpha):
        feature_4096 = self.feature(input_data)
        
        feature = self.bottleneck(feature_4096)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        
        
        domain_output_st = self.dc_ip3_st(self.domain_classifier(reverse_feature))   
        domain_output_mt = self.dc_ip3_mt(self.domain_classifier(reverse_feature))

        return class_output, domain_output_st,domain_output_mt,feature_4096 



class Alex_Model_OfficeHome(nn.Module):

    def __init__(self):
        super(Alex_Model_OfficeHome, self).__init__()
        self.feature = nn.Sequential()
        #self.feature.add_module('instance_norm_1', nn.InstanceNorm2d((30,3,227,227)))
        self.feature.add_module('conv1', nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0))       
        self.feature.add_module('relu1', nn.ReLU(True))
        self.feature.add_module('norm1', LRN(local_size=5, alpha=0.0001, beta=0.75))
        self.feature.add_module('pool1', nn.MaxPool2d(kernel_size=3, stride=2)) 
        
        self.feature.add_module('conv2', nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2))       
        self.feature.add_module('relu2', nn.ReLU(True))
        self.feature.add_module('norm2', LRN(local_size=5, alpha=0.0001, beta=0.75))
        self.feature.add_module('pool2', nn.MaxPool2d(kernel_size=3, stride=2))  
        
        self.feature.add_module('conv3', nn.Conv2d(256, 384, kernel_size=3, padding=1))       
        self.feature.add_module('relu3', nn.ReLU(True))
        
        self.feature.add_module('conv4', nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2))       
        self.feature.add_module('relu4', nn.ReLU(True))
        
        self.feature.add_module('conv5', nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2))       
        self.feature.add_module('relu5', nn.ReLU(True))
        self.feature.add_module('pool5', nn.MaxPool2d(kernel_size=3, stride=2)) 
  
        self.feature.add_module('view',View(-1, 256 * 6 * 6)) 
        self.feature.add_module('fc6',nn.Linear(256 * 6 * 6, 4096))
        self.feature.add_module('relu6',nn.ReLU(inplace=True))
        self.feature.add_module('drop6', nn.Dropout())
        self.feature.add_module('fc7',nn.Linear(4096, 4096))
        self.feature.add_module('relu7',nn.ReLU(inplace=True))
        self.feature.add_module('drop7', nn.Dropout())
        self.bottleneck = nn.Linear(4096, 256)
        
        
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('fc8',nn.Linear(256, 65))
    
        

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('dc_ip1', nn.Linear(256 , 1024))  
        self.domain_classifier.add_module('dc_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('dc_drop1', nn.Dropout())
        self.domain_classifier.add_module('dc_ip2', nn.Linear(1024, 1024))  
        self.domain_classifier.add_module('dc_relu2', nn.ReLU(True))
        
        self.dc_ip3_st = nn.Linear(1024, 1)
        
        
        self.dc_ip3_mt = nn.Linear(1024, 3)
        

    def forward(self, input_data, alpha):
        feature4096 = self.feature(input_data)
        
        feature = self.bottleneck(feature4096)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output_st = self.dc_ip3_st(self.domain_classifier(reverse_feature))       
        domain_output_mt = self.dc_ip3_mt(self.domain_classifier(reverse_feature))


        return class_output, domain_output_st,domain_output_mt,feature4096        