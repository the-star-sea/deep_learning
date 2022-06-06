from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.nn as nn

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size

        self.hx = nn.Linear(input_dim, hidden_dim)
        self.hh = nn.Linear(hidden_dim, hidden_dim)
        self.oh = nn.Linear(hidden_dim, output_dim)


    def forward(self, x):
        # Implementation here ...

        x = x.unsqueeze(2)
        if torch.cuda.is_available():
            h = torch.zeros((self.hidden_dim, self.hidden_dim), dtype=torch.float32).cuda()
        else:
            h= torch.zeros((self.hidden_dim, self.hidden_dim), dtype=torch.float32)
        for i in range(self.seq_length):
            h = torch.tanh(self.hx(x[:, i]) + self.hh(h))

        o = self.oh(h)
        out = F.softmax(o, dim=1)

        return out

