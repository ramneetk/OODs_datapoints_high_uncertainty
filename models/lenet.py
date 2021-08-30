import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        
        self.layer1 = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh())
        
        self.layer2 = nn.AvgPool2d(kernel_size=2)

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh())
        
        self.layer4 = nn.AvgPool2d(kernel_size=2)

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh())

        self.layer6 =  nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh())
        
        self.layer7 =  nn.Linear(in_features=84, out_features=n_classes)

        self.all_layers = [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7]


    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        logits = self.layer7(x)
        return logits
    
    def intermediate_forward(self,x,layer_num): # for penultimate forward, layer_index = 5
        for i in range(layer_num+1):
            x = self.all_layers[i](x)
        return x
    
    def feature_list(self, x):
        out_list = []
        x = self.layer1(x)
        #out_list.append(x)
        x = self.layer2(x)
        #out_list.append(x)
        x = self.layer3(x)
        #out_list.append(x)
        x = self.layer4(x)
        #out_list.append(x)
        x = self.layer5(x)
        #out_list.append(x)
        x = self.layer6(x)
        out_list.append(x)
        out = self.layer7(x)
        return out, out_list