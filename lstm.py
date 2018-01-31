import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

##input = 42126x 525 tensor

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch, wordveclen):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch = batch
        self.wordidxlen = wordidxlen
        
        ##self.h_0 = Variable(torch.Tensor(self.num_layers, self.batch, self.hidden_size).uniform_(-0.05,0.05))
        ##self.c_0 = Variable(torch.Tensor(self.num_layers, self.batch, self.hidden_size).uniform_(-0.05,0.05))
        self.lstm = torch.nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size, 
                num_layers = self.num_layers, batch_first = True, dropout=0.5)
        self.fc = torch.nn.Linear(self.hidden_size, self.wordidxlen)
        self.softmax = torch.nn.softmax()

    ##def init_parameters(self):
        ##self.


    def forward(self,X,HC):
        ##h_0 = self.h_0
        ##c_0 = self.c_0

        output, (h_T, c_T) = self.lstm(X,HC)
        ##output = output.contiguous()
        out = self.fc(output)
        out = self.softmax(out)
        return out, h_T, c_T
        







		
