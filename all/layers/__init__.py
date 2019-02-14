import torch
from torch import nn

class Aggregation(nn.Module):
    '''
    Aggregation layer for the Dueling architecture.

    https://arxiv.org/abs/1511.06581
    This layer computes a Q function by combining
    an estimate of V with an estimate of the advantage.
    The advantage is normalized by substracting the average
    advantage so that we can propertly 
    '''
    def forward(self, value, advantages):
        return value + advantages - torch.mean(advantages, dim=1, keepdim=True)

class Dueling(nn.Module):
    '''
    Implementation of the head for the Dueling architecture.

    https://arxiv.org/abs/1511.06581
    This module computes a Q function by computing
    an estimate of V, and estimate of the advantage,
    and combining them with a special Aggregation layer.
    '''
    def __init__(self, value_model, advantage_model):
        super(Dueling, self).__init__()
        self.value_model = value_model
        self.advantage_model = advantage_model
        self.aggregation = Aggregation()

    def forward(self, features):
        value = self.value_model(features)
        advantages = self.advantage_model(features)
        return self.aggregation(value, advantages)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

__all__ = ["Aggregation", "Dueling", "Flatten"]
