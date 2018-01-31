import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import random

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = torch.nn.Sequential(
                torch.nn.Conv2d(1,25,(1,15)),
                torch.nn.MaxPool2d(kernel_size = (21,1))
                )
        self.conv2 = torch.nn.Sequential(
                torch.nn.Conv2d(1,50,(2,15)),
                torch.nn.MaxPool2d(kernel_size = (20,1))
                )
        self.conv3 = torch.nn.Sequential(
                torch.nn.Conv2d(1,75,(3,15)),
                torch.nn.MaxPool2d(kernel_size = (19,1))
                )
        self.conv4 = torch.nn.Sequential(
                torch.nn.Conv2d(1,100,(4,15)),
                torch.nn.MaxPool2d(kernel_size = (18,1))
                )
        self.conv5 = torch.nn.Sequential(
                torch.nn.Conv2d(1,125,(5,15)),
                torch.nn.MaxPool2d(kernel_size = (17,1))
                )
        self.conv6 = torch.nn.Sequential(
                torch.nn.Conv2d(1,150,(6,15)),
                torch.nn.MaxPool2d(kernel_size = (16,1)
                )
            )
        self.Tanh = torch.nn.Tanh()
        ##self.add_bias = torch.nn.Add(inputDimension = (25+50+75+100+125+150))
    def forward(self, X):
        conv1out = self.conv1(X)
        ##print("conv1out")
        ##print(conv1out)
        ##print(conv1out.size())
        conv2out = self.conv2(X)
        conv3out = self.conv3(X)
        conv4out = self.conv4(X)
        conv5out = self.conv5(X)
        conv6out = self.conv6(X)
        ##print("conv6out")
        ##print(conv6out)
        ##print(conv6out.size())
        conv1out = conv1out.squeeze(3)
        conv2out = conv2out.squeeze(3)
        conv3out = conv3out.squeeze(3)
        conv4out = conv4out.squeeze(3)
        conv5out = conv5out.squeeze(3)
        conv6out = conv6out.squeeze(3)
        conv1out = conv1out.squeeze(2)
        conv2out = conv2out.squeeze(2)
        conv3out = conv3out.squeeze(2)
        conv4out = conv4out.squeeze(2)
        conv5out = conv5out.squeeze(2)
        conv6out = conv6out.squeeze(2)

        conv_concat = torch.cat((conv1out, conv2out, conv3out, conv4out, conv5out, conv6out),1)
        ##out = add_bias(conv_concat)
        out = self.Tanh(conv_concat)
        return out
    
    ##train the model		
    
    




