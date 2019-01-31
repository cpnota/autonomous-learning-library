import torch
from torch import nn


class QTable(nn.Module):
    def __init__(self, states, actions):
        super(QTable, self).__init__()
        self.values = nn.Parameter(torch.zeros(states, actions))

    def forward(self, state):
        return self.values[state]

class VTable(nn.Module):
    def __init__(self, states):
        super(VTable, self).__init__()
        self.values = nn.Parameter(torch.zeros(states))

    def forward(self, state):
        return self.values[state].unsqueeze(-1)
