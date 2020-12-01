import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from trigram import getVector

class LSTM(nn.Module):
    def __init__(self,embedding_dim, hidden_dim, vocab_size, encode_dim, label_size, batch_size, use_gpu=True, pretrained_weight=None):
        super(LSTM,self).__init__()
        self.vector = None
        self.embeddings = nn.Embedding(vocab_size,embedding_dim)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.gpu = use_gpu
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.batch_size = batch_size
        self.th = torch.cuda if use_gpu else torch
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, dropout=0.2)
        self.hidden = nn.Linear(self.hidden_dim, self.label_size)
        if pretrained_weight is not None:
            self.embeddings.weight.data.copy_(torch.from_numpy(pretrained_weight))
    
    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.embedding_dim))
        if self.gpu:
            return zeros.cuda()
        return zeros

    def padding(self, x):
        lens = [len(item) for item in x]
        max_len = max(lens)
        for i in range(len(lens)):
            if max_len-lens[i]:
                x[i].extend([0] * (max_len-lens[i]))
        return Variable(self.th.LongTensor(x)), lens#Variable(self.th.LongTensor(lens))
    
    def forward(self,x1,x2):
        x1, lengths1 = self.padding(x1)
        x2, lengths2 = self.padding(x2)
        temp1 = self.embeddings(x1)
        temp2 = self.embeddings(x2)
        x1, (h1, c1) = self.lstm(temp1)
        x2, (h2, c2) = self.lstm(temp2)
        x1 = torch.transpose(x1, 1, 2)
        x2 = torch.transpose(x2, 1, 2)
        x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)
        abs_dist = torch.abs(torch.add(x1,-x2))
        y = torch.sigmoid(self.hidden(abs_dist))
        return y

class BiLSTM(nn.Module):
    def __init__(self,embedding_dim, hidden_dim, vocab_size, encode_dim, label_size, batch_size, use_gpu=True, pretrained_weight=None):
        super(BiLSTM,self).__init__()
        self.embeddings = nn.Embedding(vocab_size,embedding_dim)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.gpu = use_gpu
        self.vocab_size = vocab_size
        self.label_size = label_size
        self.batch_size = batch_size
        self.th = torch.cuda if use_gpu else torch
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, dropout=0.2,bidirectional=True)
        self.hidden = nn.Linear(self.hidden_dim*2, self.label_size)
        if pretrained_weight is not None:
            self.embeddings.weight.data.copy_(torch.from_numpy(pretrained_weight))
    
    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.embedding_dim))
        if self.gpu:
            return zeros.cuda()
        return zeros

    def padding(self, x):
        lens = [len(item) for item in x]
        max_len = max(lens)
        for i in range(len(lens)):
            if max_len-lens[i]:
                x[i].extend([0] * (max_len-lens[i]))
        return Variable(self.th.LongTensor(x)), lens#Variable(self.th.LongTensor(lens))
    
    def forward(self,x1,x2):
        x1, lengths1 = self.padding(x1)
        x2, lengths2 = self.padding(x2)
        temp1 = self.embeddings(x1)
        temp2 = self.embeddings(x2)
        x1, (h1, c1) = self.lstm(temp1)
        x2, (h2, c2) = self.lstm(temp2)
        x1 = torch.transpose(x1, 1, 2)
        x2 = torch.transpose(x2, 1, 2)
        x1 = F.max_pool1d(x1, x1.size(2)).squeeze(2)
        x2 = F.max_pool1d(x2, x2.size(2)).squeeze(2)
        abs_dist = torch.abs(torch.add(x1,-x2))
        y = torch.sigmoid(self.hidden(abs_dist))
        return y

class DNN(nn.Module):
    def __init__(self,label_size,batch_size,use_gpu=True):
        super(DNN,self).__init__()
        self.gpu = use_gpu
        self.label_size = label_size
        self.batch_size = batch_size
        self.th = torch.cuda if use_gpu else torch
        self.subnetwork1 = nn.Linear(9,100)
        self.subnetwork2 = nn.Linear(100,100)
        self.comparator1 = nn.Linear(100,50)
        self.comparator2 = nn.Linear(50,10)
        self.comparator3 = nn.Linear(10,self.label_size)
        self.dropout = torch.nn.Dropout(p=0.2)
    def forward(self,x1,x2):
        x1 = self.subnetwork1(Variable(self.th.FloatTensor(x1)))
        x2 = self.subnetwork1(Variable(self.th.FloatTensor(x2)))
        x1 = self.subnetwork2(x1)
        x2 = self.subnetwork2(x2)
        abs_dist = torch.abs(torch.add(x1,-x2))
        y = self.comparator1(abs_dist)
        y = self.comparator2(y)
        y = self.comparator3(y)
        y = torch.sigmoid(y)
        return y