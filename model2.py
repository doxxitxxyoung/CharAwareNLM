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
        self.conv1 = torch.nn.Conv2d(1,25,(1,15))
        self.bias1 = torch.nn.Parameter(torch.ones(700,25,21))
        self.max1 = torch.nn.MaxPool1d(kernel_size =21)
        self.conv2 = torch.nn.Conv2d(1,50,(2,15))
        self.bias2 = torch.nn.Parameter(torch.ones(700,50,20))
        self.max2 = torch.nn.MaxPool1d(kernel_size =20)
        self.conv3 = torch.nn.Conv2d(1,75,(3,15))
        self.bias3 = torch.nn.Parameter(torch.ones(700,75,19))
        self.max3 = torch.nn.MaxPool1d(kernel_size =19)
        self.conv4 = torch.nn.Conv2d(1,100,(4,15))
        self.bias4 = torch.nn.Parameter(torch.ones(700,100,18))
        self.max4 = torch.nn.MaxPool1d(kernel_size =18)
        self.conv5 = torch.nn.Conv2d(1,125,(5,15))
        self.bias5 = torch.nn.Parameter(torch.ones(700,125,17))
        self.max5 = torch.nn.MaxPool1d(kernel_size =17)
        self.conv6 = torch.nn.Conv2d(1,150,(6,15))
        self.bias6 = torch.nn.Parameter(torch.ones(700,150,16))
        self.max6 = torch.nn.MaxPool1d(kernel_size =16)

        self.tanh = torch.nn.Tanh()

        self.input_size = input_size

        self.trans_gate1 = torch.nn.Linear(in_features = self.input_size, out_features = self.input_size, bias = True)
        self.trans_gate2 = torch.nn.Sigmoid()

        self.g_Wyb1 = torch.nn.Linear(in_features = self.input_size, out_features = self.input_size, bias = True)
        self.g_Wyb2 = torch.nn.ReLU()

        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch = batch
        self.wordidxlen = wordidxlen


        self.lstm = torch.nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, 
                num_layers = self.num_layers, batch_first = True, dropout=self.dropout_rate)
        self.dropout1 = torch.nn.Dropout(p=self.dropout_rate)
        self.dropout2 = torch.nn.Dropout(p=self.dropout_rate)
        self.dropout3 = torch.nn.Dropout(p=self.dropout_rate)

        #self.dropout_ = torch.nn.Dropout()
        self.outputembed = torch.nn.Linear(self.hidden_size, self.wordidxlen, bias= True)  #bias=true?????
        #self.biasq = torch.nn.Parameter(torch.ones(700,125,17))

        self.softmax = torch.nn.Softmax(dim=1)

        #self.add_bias = torch.nn.Add(inputDimension = (25+50+75+100+125+150))

    def weight_init(self):
        self.conv1.weight.data.uniform_(-0.05,0.05)
        self.conv2.weight.data.uniform_(-0.05,0.05)
        self.conv3.weight.data.uniform_(-0.05,0.05)
        self.conv4.weight.data.uniform_(-0.05,0.05)
        self.conv5.weight.data.uniform_(-0.05,0.05)
        self.conv6.weight.data.uniform_(-0.05,0.05)

        self.trans_gate1.weight.data.uniform_(-0.05,0.05)
        self.trans_gate1.bias.data.uniform_(-2.05,-1.95)
        self.g_Wyb1.weight.data.uniform_(-0.05,0.05)
        self.lstm.weight_ih_l0.data.uniform_(-0.05,0.05)
        self.lstm.weight_hh_l0.data.uniform_(-0.05,0.05)
        self.lstm.weight_ih_l1.data.uniform_(-0.05,0.05)
        self.lstm.weight_hh_l1.data.uniform_(-0.05,0.05)
        self.outputembed.weight.data.uniform_(-0.05,0.05)


    def forward(self, X, HC):
        ##### Embedding
        embed = self.embedding(X)  #700x25 -> 700x25x15
        embed = embed.unsqueeze(1) #700x1x21x15
        ##print(self.embedding.weight.grad) ##returns 0

        ##### CNN 
        #conv1out = self.max1(self.conv1(embed).squeeze(3))
        #conv2out = self.max2(self.conv2(embed).squeeze(3))
        #conv3out = self.max3(self.conv3(embed).squeeze(3))
        #conv4out = self.max4(self.conv4(embed).squeeze(3))
        #conv5out = self.max5(self.conv5(embed).squeeze(3))
        #conv6out = self.max6(self.conv6(embed).squeeze(3))
        conv1out = self.max1(self.tanh(self.conv1(embed).view(700,25,21)+self.bias1)).view(700,25)
        conv2out = self.max2(self.tanh(self.conv2(embed).view(700,50,20)+self.bias2)).view(700,50)
        conv3out = self.max3(self.tanh(self.conv3(embed).view(700,75,19)+self.bias3)).view(700,75)
        conv4out = self.max4(self.tanh(self.conv4(embed).view(700,100,18)+self.bias4)).view(700,100)
        conv5out = self.max5(self.tanh(self.conv5(embed).view(700,125,17)+self.bias5)).view(700,125)
        conv6out = self.max6(self.tanh(self.conv6(embed).view(700,150,16)+self.bias6)).view(700,150)

        #conv1out = conv1out.squeeze(2)
        #conv2out = conv2out.squeeze(2)
        #conv3out = conv3out.squeeze(2)
        #conv4out = conv4out.squeeze(2)
        #conv5out = conv5out.squeeze(2)
        #conv6out = conv6out.squeeze(2)
        #conv1out = conv1out.view(700,25)
        #conv2out = conv2out.view(700,50)
        #conv3out = conv3out.view(700,75)
        #conv4out = conv4out.view(700,100)
        #conv5out = conv5out.view(700,125)
        #conv6out = conv6out.view(700,150)
        out = torch.cat((conv1out, conv2out, conv3out, conv4out, conv5out, conv6out),1)
        #out = self.Tanh(conv_concat+self.bias) #700x525
        #print("tanhout")
        #print(out)

        ##### Highway network
        T = self.trans_gate1(out)
        T = self.trans_gate2(T)
        #print("T")
        #print(T)
        Wyb = self.g_Wyb1(out)
        Wyb = self.g_Wyb2(Wyb)
        #print("Wyb")
        #print(Wyb)
        out = T*Wyb+(1-T)*out
        #out = torch.add(torch.mul(T,Wyb),torch.mul((1-T),out)) #700x525)
        #print("highway")
        #out = self.outputembed(out) #700x9998#print(out)
        out = out.view(20,35,-1) #20x35x525
        #print(out)

        #####LSTM
        out, hc = self.lstm(out,HC)  #20x35x300 , 20x300
        out = self.dropout1(out)
        #H = hc[0]
        #C = hc[1]
        #H = self.dropout2(H)
        #C = self.dropout3(C)
        #hc = (H,C)
        #out = self.dropout_out(out)
        #print("lstm1")
        #print(out)
        #print(h)
        out = out.view(700,self.hidden_size) #700x300
        #print("lstm")
        #print(out)
        out = self.outputembed(out)
        out = self.softmax(out)
        
        #probout = self.softmax(torch.matmul(H,out)+self.biasq)
        return out, hc
