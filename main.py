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

from cnn import CNN
from highway import Highway
from lstm import LSTM



#hyper parameters
batch_size = 20
backprop_steps = 35
learning_rate = 1.0 #to 0.5

#parameters are randomly initialized as []

"""BATCH
torch_dataset = Data.TensorDataset(data_tensor=x, target_tensor=y)
loader = Data.DataLoader(
    dataset=torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=2,              # subprocesses for loading data
)

for epoch in range(3):   # train entire dataset 3 times
    for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
        # train your data...
        print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',
              batch_x.numpy(), '| batch y: ', batch_y.numpy())            
              """

##train_dataset = open("C:/Users/DoYeong Hwang/Desktop/CharAwareNLM/data/ptb/train.txt","r", encoding = 'utf-8')
train_dataset = open("/home/newcome/dyh/CharAwareNLM/data/ptb/train.txt","r", encoding = 'utf-8')
"""
train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
        batch_size = 20, shuffle = True, num_workers=1)
for batch in train_loader:
    print(batch)
"""
train_data = train_dataset.read() #string\
torch.manual_seed(1)

##learning
##learn = parser.add_argument_group('learning options')
##learn.add_argument('--lr', type=float, default = 1.0, help='initial learning rate [default: 1.0]')
##learn.add_argument(
##learn.add_argument(
##learn.add_argument(


#data
##parser.add_argument('-shuffle', action = 'store_true', default = False, help = 'shuffle the data every epoch')


#HyperParameters
##parser.add_argument('-dropout', type = float, default = 0.5, help='prob of dropout')
##print(len(train_data))
##print(len(train_data.split())) 842501
max_lan = len(max(train_data.split(),key=len))
##padding_words = '@'*max_lan
##print(len(set(train_data)))
##print(padding_words)
train_words = train_data.split()
total_words = len(train_words) ##887521 words
##train_words = train_data.split() + [padding_words]*19  ##19is 842800-842501



max_lan = len(max(train_words,key=len))
print(max_lan)

def paddingB(train_words):
    max_lan = len(max(train_words,key=len))
    for i in range(len(train_words)):
            l = len(train_words[i])
            if l < max_lan:
                train_words[i] = train_words[i]+ ''.join(['@'for num in range(max_lan-l)])

paddingB(train_words) ##19chars


def paddingA(train_words):
    for i in range(len(train_words)):
        train_words[i] = train_words[i].join(('S','E'))
        


paddingA(train_words) ##maximum 21 chars


print(train_words[35])


xwords = train_words[0:total_words-1] ##887520 words
ywords = train_words[1:total_words]   ##887520 words
padding_words = '@'*max_lan
xwords = xwords + ['S'+padding_words+'E']*80  ##19is 887600-887520
ywords = ywords.view(20,35,-1)
print(ywords.size())

#####words indexing
word_set = set(train_words)
word_to_ix = {word: i for i,word in enumerate(word_set)}
wordidxlen = len(word_set)

#ywordidx = [word_to_idx[i] for i in ywords]
#print("ywordidx")
#print(ywordidx)
#print(len(ywordidx))



#####x words char embedding

train_char = set(train_data) ##50 chars
##print(len(train_char))
train_char.update('@','S','E') ###53 chars

dim_char_emb = 15
char_to_ix = {char: i for i, char in enumerate(train_char)}
[char_to_ix['@']]
embeds = nn.Embedding(len(train_char),15, padding_idx=[char_to_ix['@']])


def input_var(xwords):
    x_size = len(xwords)
    char_embed = Variable(torch.zeros(x_size, 21, 15))
    for j in range(x_size):
        idx = [char_to_ix[k] for k in xwords[j]]
        print(idx)
        char_embed[j] = embeds(Variable(torch.LongTensor(idx)))
        return char_embed
#batch 에 맞게 iter는 나중에 다시
char_embed= input_var(xwords)
##print(char_embedding_i)
##print(char_embedding_i.size())
char_embed = char_embed.unsqueeze(1)
##print("char_embedding_i[0]")
##print(char_embedding_i[0])
##print(char_embedding_i.size())
##print(char_embed.size()) ##887520x1x21x15
batch_num = 20
seq_len = 35
char_embed = char_embed.view(batch_num,-1,1,21,15) ##20x44376x1x21x15
#print(char_embed.size())

##torch.cuda.is_available()
##torch.cuda.max_memory_allocated()
### CNN
###print(torch.cuda.is_available())
"""
m = torch.nn.Conv2d(1,75,(3,15))
n = torch.nn.MaxPool2d((19,1))
print(n(m(char_embedding_i)).size())
x=n(m(char_embedding_i))
x = x.squeeze(2,3)
print(x.size())
"""
##cnn = CNN()
learning_rate = 1.0
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(lstm.parameters(), lr = learning_rate)


for i in range(0, 44376-1, 35):
    #print("charembed")
    #print(char_embed[:,i:i+35,:])
    #print(char_embed[:,i:i+35,:].size())
    cnn = CNN()
    char_embedx = char_embed[:,i:i+35,:].contiguous().view(-1,1,21,15)
    #print("char_embedx")
    #print(char_embedx)
    #print(char_embedx.size())
    cnnoutput = cnn(char_embedx)
    #print('CNN')
    #print(cnnoutput)
    #print(cnnoutput.size())

    input_size = cnnoutput.size()[1]
    output_size = cnnoutput.size()[1]
    highway = Highway(input_size, output_size)
    highwayoutput = highway(cnnoutput)
    #print("highway")
    #print(highwayoutput)
    #print(highwayoutput.size())

    hidden_size = 300
    num_layers = 2
    batch = 20
    lstm = LSTM(input_size,hidden_size,num_layers,batch)
    lstm.zero_grad()
    lstminput = highwayoutput.view(20,35,-1)
    print("lstminput")
    print(lstminput)
    print(lstminput.size())
    H = Variable(torch.randn(num_layers, batch, hidden_size).uniform_(-0.05,0.05))
    C = Variable(torch.randn(num_layers, batch, hidden_size).uniform_(-0.05,0.05))
    HC = (H,C)
    lstmoutput, ht, ct = lstm(lstminput,HC)
    ywordx = ywords[:,i:i+35,:]
    ywordsidx = Variable(word_to_idx[
    print("lstm")
    print(lstmoutput)
    print("ht")
    print(ht)
    print("ct")
    print(ct)
    loss = criterion(lstmoutput, 
    








"""
for i in range(batch_size):
    char_embed[i]=cnn(char_embed[i])
"""
##print(cnnoutput) #842520 x 525
#cnnoutput = cnnoutput.view(20,1204,35,525)
##print(cnnoutput)
##print(cnnoutput.size())
#print(cnnoutput)
#print(cnnoutput.size())
#x = cnnoutput.size()
#print(x)

##Highway
"""
input_size = cnnoutput.size()[1]
output_size = cnnoutput.size()[1]
highway = Highway(input_size, output_size)
highwayoutput = highway(cnnoutput)
print(highwayoutput) #842520 x 525
"""
##LSTM
"""
def seq_batch(Q):
    Q1 = Q.view(20,-1)#20x42126x525
    for i in len
"""
"""
input_size = highwayoutput.size()[1]
hidden_size = 300
num_layers = 2
"""
##drop out 0.5
## L2 grad constraint = 5
##parameters of the models are init with -0.05,0.05



""""
model = torch.nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, 
        batch_first = True, dropout=0.5)
input = highwayoutput.reshape(20,1204,-1)
print(input)
"""

