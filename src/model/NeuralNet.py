import torch
import torch.nn as nn

# Model
class NeuralNet(nn.Module):
    def __init__(self, inpt, hidn, oupt):
        super(NeuralNet, self).__init__()

        self.dropout = nn.Dropout(p=0.2)
        self.act = nn.ReLU()

        self.l1 = nn.Linear(inpt, hidn)
        self.l2 = nn.Linear(hidn, oupt)

    def forward(self, x):
        out = self.l1(x.to(torch.float32))
        out = self.dropout(out)
        out = self.act(out)
        out = self.l2(out)
        out = self.act(out)

        return out
# NueralNet