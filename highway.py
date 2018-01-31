
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
class Highway(nn.Module):
    def __init__(self, input_size, output_size):
        super(Highway, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.transform_gate = torch.nn.Sequential(
                torch.nn.Linear(in_features = self.input_size,
                    out_features = self.input_size,
                    bias = True),
                torch.nn.Sigmoid()
                )

        self.g_Wyb = torch.nn.Sequential(
                torch.nn.Linear(in_features = self.input_size,
                    out_features = self.output_size,
                    bias = True),
                torch.nn.ReLU()
                )
## sonething nonlinear functions and combine togethe
    def forward(self,Y):
        T = self.transform_gate(Y)
        Wyb = self.g_Wyb(Y)
        Z = T*Wyb + (1-T)*Y
        ##Z = self.transform_gate(Y)*self.g_Wyb(Y) + (1-self.transform_gate(Y))*Y
        return Z

