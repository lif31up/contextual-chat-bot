import numpy as np
import torch
import torch.nn as nn


# Model
class NeuralNet(nn.Module):
    def __init__(self, inpt, hidn, oupt):
        super(NeuralNet, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.act = nn.ReLU()

        self.l1 = nn.Linear(inpt, hidn)
        self.l2 = nn.Linear(hidn, hidn)
        self.l3 = nn.Linear(hidn, hidn)
        self.l4 = nn.Linear(hidn, oupt)

    def forward(self, x):
        out = self.l1(x.to(torch.float32))
        out = self.act(out)
        out = self.l2(out)
        out = self.act(out)
        out = self.l3(out)
        out = self.dropout(out)
        out = self.act(out)
        out = self.l4(out)
        out = self.act(out)

        return out
