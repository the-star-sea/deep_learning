from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(LSTM, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size


        # Linear Layers
        self.w_gx = nn.Linear(input_dim, hidden_dim)
        self.w_gh = nn.Linear(hidden_dim, hidden_dim)
        self.w_ix = nn.Linear(input_dim, hidden_dim)
        self.w_ih = nn.Linear(hidden_dim, hidden_dim)
        self.w_fx = nn.Linear(input_dim, hidden_dim)
        self.w_fh = nn.Linear(hidden_dim, hidden_dim)
        self.w_ox = nn.Linear(input_dim, hidden_dim)
        self.w_oh = nn.Linear(hidden_dim, hidden_dim)
        self.w_ph = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(2)
        # parameters


        # use for to forward
        if torch.cuda.is_available():
            self.h = torch.zeros(self.hidden_dim, self.hidden_dim).cuda()
            self.c = torch.zeros(self.hidden_dim, self.hidden_dim).cuda()
            self.bg = torch.zeros(self.batch_size, self.batch_size).cuda()
            self.bi = torch.zeros(self.batch_size, self.batch_size).cuda()
            self.bf = torch.zeros(self.batch_size, self.batch_size).cuda()
            self.bo = torch.zeros(self.batch_size, self.batch_size).cuda()
            self.bp = torch.zeros(self.batch_size, self.output_dim).cuda()
        else:
            self.h = torch.zeros(self.hidden_dim, self.hidden_dim)
            self.c = torch.zeros(self.hidden_dim, self.hidden_dim)
            self.bg = torch.zeros(self.batch_size, self.batch_size)
            self.bi = torch.zeros(self.batch_size, self.batch_size)
            self.bf = torch.zeros(self.batch_size, self.batch_size)
            self.bo = torch.zeros(self.batch_size, self.batch_size)
            self.bp = torch.zeros(self.batch_size, self.output_dim)
        for t in range(self.seq_length):
            temp = self.w_gx(x[:, t]) + self.w_gh(self.h) + self.bg
            g = torch.tanh(temp)
            temp = self.w_ix(x[:, t]) + self.w_ih(self.h) + self.bi
            i = torch.sigmoid(temp)
            temp = self.w_fx(x[:, t]) + self.w_fh(self.h) + self.bf
            f = torch.sigmoid(temp)
            temp = self.w_ox(x[:, t]) + self.w_oh(self.h) + self.bo
            o = torch.sigmoid(temp)
            self.c = self.c * f + g * i
            self.h = torch.tanh(self.c) * o

        self.p = self.w_ph(self.h) + self.bp
        out = F.softmax(self.p, dim=1)
        return out
        
