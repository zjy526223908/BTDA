import torch.nn as nn
import torch
from functions import ReverseLayerF

class noise(nn.Module):
    def __init__(self, std):
        super(noise, self).__init__()
        self.std = std
    def forward(self, input):
        eps = torch.nn.init.normal_(input, mean=0, std=self.std)
        input = input+eps
        return input
class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, input):
        return input.view(*self.shape)
        
class Model(nn.Module):

    def __init__(self,is_in, batch_size):
        super(Model, self).__init__()      
        self.feature = nn.Sequential()                   #input 3*32*32
        if is_in==1:
            self.feature.add_module('instance_norm_1', nn.InstanceNorm2d((batch_size,3,32,32)))
        self.feature.add_module('conv1_1', nn.Conv2d(3, 64, kernel_size=3,stride=1,padding=1))  #64*32*32
        self.feature.add_module('bn1_1', nn.BatchNorm2d(64))
        self.feature.add_module('relu1_1', nn.LeakyReLU(0.1))
        self.feature.add_module('conv1_2', nn.Conv2d(64, 64, kernel_size=3,stride=1,padding=1)) #64*32*32 
        self.feature.add_module('bn1_2', nn.BatchNorm2d(64))
        self.feature.add_module('relu1_2', nn.LeakyReLU(0.1))
        self.feature.add_module('conv1_3', nn.Conv2d(64, 64, kernel_size=3,stride=1,padding=1)) #64*32*32
        self.feature.add_module('bn1_3', nn.BatchNorm2d(64))
        self.feature.add_module('relu1_3', nn.LeakyReLU(0.1))
        self.feature.add_module('pool1', nn.MaxPool2d(kernel_size=2,stride=2))                  #64*16*16
        self.feature.add_module('drop1', nn.Dropout2d(0.5))
        #self.feature.add_module('noise1', noise(1))
        
        self.feature.add_module('conv2_1', nn.Conv2d(64, 64, kernel_size=3,stride=1,padding=1)) #64*16*16
        self.feature.add_module('bn2_1', nn.BatchNorm2d(64))
        self.feature.add_module('relu2_1', nn.LeakyReLU(0.1))
        self.feature.add_module('conv2_2', nn.Conv2d(64, 64, kernel_size=3,stride=1,padding=1)) #64*16*16
        self.feature.add_module('bn2_2', nn.BatchNorm2d(64))
        self.feature.add_module('relu2_2', nn.LeakyReLU(0.1))
        self.feature.add_module('conv2_3', nn.Conv2d(64, 64, kernel_size=3,stride=1,padding=1)) #64*16*16 
        self.feature.add_module('bn2_3', nn.BatchNorm2d(64))
        self.feature.add_module('relu2_3', nn.LeakyReLU(0.1))
        self.feature.add_module('pool2', nn.MaxPool2d(kernel_size=2,stride=2))                  #64*8*8
        self.feature.add_module('drop2', nn.Dropout2d(0.5))
        #self.feature.add_module('noise2', noise(1))    

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('conv1_1', nn.Conv2d(64, 64, kernel_size=3,stride=1,padding=1))  
        self.class_classifier.add_module('bn1_1', nn.BatchNorm2d(64))
        self.class_classifier.add_module('relu1_1', nn.LeakyReLU(0.1))
        self.class_classifier.add_module('conv1_2', nn.Conv2d(64, 64, kernel_size=3,stride=1,padding=1))  
        self.class_classifier.add_module('bn1_2', nn.BatchNorm2d(64))
        self.class_classifier.add_module('relu1_2', nn.LeakyReLU(0.1))
        self.class_classifier.add_module('conv1_3', nn.Conv2d(64, 64, kernel_size=3,stride=1,padding=1))  
        self.class_classifier.add_module('bn1_3', nn.BatchNorm2d(64))
        self.class_classifier.add_module('relu1_3', nn.LeakyReLU(0.1))
        self.class_classifier.add_module('global_average_pool1', nn.AdaptiveAvgPool2d((1,1)))
        self.class_classifier.add_module('view',View(-1, 64 * 1 * 1)) 
        self.class_classifier.add_module('fc1', nn.Linear(64, 10))
        
    def forward(self, input_data):
        input_data = input_data.expand(input_data.data.shape[0], 3, 32, 32)
        feature_output = self.feature(input_data)
        out = self.class_classifier(feature_output)
        
        return feature_output ,out

class Class_Classifier(nn.Module):

    def __init__(self):
        super(Class_Classifier, self).__init__()
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('conv1_1', nn.Conv2d(64, 64, kernel_size=3,stride=1,padding=1))  
        self.class_classifier.add_module('bn1_1', nn.BatchNorm2d(64))
        self.class_classifier.add_module('relu1_1', nn.LeakyReLU(0.1))
        self.class_classifier.add_module('conv1_2', nn.Conv2d(64, 64, kernel_size=3,stride=1,padding=1))  
        self.class_classifier.add_module('bn1_2', nn.BatchNorm2d(64))
        self.class_classifier.add_module('relu1_2', nn.LeakyReLU(0.1))
        self.class_classifier.add_module('conv1_3', nn.Conv2d(64, 64, kernel_size=3,stride=1,padding=1))  
        self.class_classifier.add_module('bn1_3', nn.BatchNorm2d(64))
        self.class_classifier.add_module('relu1_3', nn.LeakyReLU(0.1))
        self.class_classifier.add_module('global_average_pool1', nn.AdaptiveAvgPool2d((1,1)))
        self.class_classifier.add_module('view',View(-1, 64 * 1 * 1)) 
        self.class_classifier.add_module('fc1', nn.Linear(64, 10))
        #self.class_classifier.add_module('softmax', nn.LogSoftmax())

    def forward(self, feature):
        out = self.class_classifier(feature)
        return out    
        

class Domain_Classifier(nn.Module):

    def __init__(self):
        super(Domain_Classifier, self).__init__()
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('view',View(-1, 64 * 8 * 8+10)) 
        self.domain_classifier.add_module('fc1', nn.Linear(64 * 8 * 8+10, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('relu1', nn.ReLU(True))
        
        
        self.fc_d1 = nn.Linear(100, 1)
        self.fc_d2 = nn.Linear(100, 4)
        
        #self.domain_classifier.add_module('sigmoid', nn.LogSigmoid())

    def forward(self, feature_e,feature_f):
        feature_e = feature_e.view(feature_e.size()[0],-1)
        feature_f = feature_f.view(feature_f.size()[0],-1)
        
        feature = torch.cat([feature_e,feature_f],1)
        #print feature.size()
        
        f_out = self.domain_classifier(feature)
        
        d1_out = self.fc_d1(f_out)
        d2_out = self.fc_d2(f_out)
        
        return d1_out,d2_out 
