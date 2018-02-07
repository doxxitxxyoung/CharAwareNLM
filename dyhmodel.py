import numpy as np
import textwrap
import math
import argparse
import datetime
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.utils.rnn as rnn_utils

class Model(nn.Module):
    def __init__(self, num_char, embed_dim, input_size, hidden_size, num_layers, batch, wordidxlen, paddingidx,dropout_rate):
        super(Model,self).__init__()
        self.paddingidx = paddingidx
        self.embedding = nn.Embedding(num_char, embed_dim, padding_idx=self.paddingidx)
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
                torch.nn.MaxPool2d(kernel_size = (16,1))
                )

        self.Tanh = torch.nn.Tanh()
        self.bias = torch.nn.Parameter(torch.ones(700,525))

        self.input_size = input_size

        self.transform_gate = torch.nn.Sequential(
                torch.nn.Linear(in_features = self.input_size,
                    out_features = self.input_size,
                    bias = True),
                torch.nn.Sigmoid()
                )

        self.g_Wyb = torch.nn.Sequential(
                torch.nn.Linear(in_features = self.input_size,
                    out_features = self.input_size,
                    bias = True),
                torch.nn.ReLU()
                )
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch = batch
        self.wordidxlen = wordidxlen


        self.lstm = torch.nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, 
                num_layers = self.num_layers, batch_first = True, dropout=self.dropout_rate)

        self.dropout1 = torch.nn.Dropout(self.dropout_rate)
        self.dropout2 = torch.nn.Dropout(self.dropout_rate)
        self.dropout3 = torch.nn.Dropout(self.dropout_rate)
        self.fc = torch.nn.Linear(self.hidden_size, self.wordidxlen)

        self.softmax = torch.nn.Softmax(dim=1)
    #def init_weight(m):
        #if isinstance(m,nn.Linear):
            #m.weight.data.uniform_(-0.05,0.05)
            #m.bias.data.uniform_(-0.05,0.05)
        #elif isinstance(m,nn.LSTM):

        #self.add_bias = torch.nn.Add(inputDimension = (25+50+75+100+125+150))
        """
    def weight_init(self):
        self.embedding.weight.data.uniform_(-0.05,0.05)
        self.conv1.weight.data.uniform_(-0.05,0.05)
        self.conv2.weight.data.uniform_(-0.05,0.05)
        self.conv3.weight.data.uniform_(-0.05,0.05)
        self.conv4.weight.data.uniform_(-0.05,0.05)
        self.conv5.weight.data.uniform_(-0.05,0.05)
        self.conv6.weight.data.uniform_(-0.05,0.05)
        #self.bias.data.uniform_(-0.05,0.05)
        #self.transform_gate.data.uniform_(-0.05,0.05)
        #self.g_Wyb.data.uniform_(-0.05,0.05)
        #self.lst
        self.lstm.weight.data.uniform_(-0.05,0.05)
        self.fc.weight.data.uniform_(-0.05,0.05)
        """

    def forward(self, X,HC):
        embed = self.embedding(X)  #700x21 -> 700x21x15
        embed = embed.unsqueeze(1) #700x1x21x15
        conv1out = self.conv1(embed)
        conv2out = self.conv2(embed)
        conv3out = self.conv3(embed)
        conv4out = self.conv4(embed)
        conv5out = self.conv5(embed)
        conv6out = self.conv6(embed)
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
        #print("conv_concat")
        #print(conv_concat)
        ##out = add_bias(conv_concat)
        out = self.Tanh(conv_concat+self.bias) #700x525
        #print("tanhout")
        #print(out)

        T = self.transform_gate(out)
        #print("T")
        #print(T)
        Wyb = self.g_Wyb(out)
        #print("Wyb")
        #print(Wyb)
        out = T*Wyb+(1-T)*out
        #out = torch.add(torch.mul(T,Wyb),torch.mul((1-T),out)) #700x525)
        #print("highway")
        #print(out)
        out = out.contiguous().view(20,35,-1) #20x35x525
        #print(out)
        out,hc  = self.lstm(out,HC)  #20x35x300
        #out = self.dropout1(out)
        """
        H = hc[0]
        C = hc[1]
        H = self.dropout2(H)
        C = self.dropout3(C)
        hc = (H,C)
        """
        #print("lstm1")
        #print(out)
        #print(hc)
        out = out.contiguous().view(700,self.hidden_size) #700x300
        #print("lstm")
        #print(out)

        out = self.fc(out) #700x10000
        out = self.softmax(out) #700 x 10000
        #print("fc")
        #print(out)
        #h_T = self.dropout_h(h_T)
        #c_T = self.dropout_c(c_T)
        #hc = (h_T,c_T)
        return out, hc
