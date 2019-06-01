import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from all.environments import State

class ListNetwork(nn.Module):
    '''
    Wraps a network such that States can be given as input.
    '''

    def __init__(self, model, out):
        super().__init__()
        self.model = model
        self.out = list(out)
        self.device = next(model.parameters()).device

    def forward(self, state):
        return self.model(state.features) * state.done.float().unsqueeze(-1)

class ListToList(nn.Module):
    '''
    Wraps a network such that States can be given as inputs, and are received as output.
    '''

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.device = next(model.parameters()).device

    def forward(self, state):
        return State(self.model(state.features), state.done, state.info)

class Aggregation(nn.Module):
    '''len()
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


class NoisyLinear(nn.Linear):
    '''
    Implementation of Linear layer for NoisyNets

    https://arxiv.org/abs/1706.10295
    NoisyNets are a replacement for epsilon greedy exploration.
    Gaussian noise is added to the weights of the output layer, resulting in
    a stochastic policy. Exploration is implicitly learned at a per-state
    and per-action level, resulting in smarter exploration.
    '''

    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.Tensor(
            out_features, in_features).fill_(sigma_init))
        self.register_buffer(
            "epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(
                torch.Tensor(out_features).fill_(sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = np.sqrt(3 / self.in_features)
        nn.init.uniform_(self.weight, -std, std)
        nn.init.uniform_(self.bias, -std, std)

    def forward(self, x):
        bias = self.bias

        if not self.training:
            return F.linear(x, self.weight, bias)

        torch.randn(self.epsilon_weight.size(), out=self.epsilon_weight)
        if self.bias is not None:
            torch.randn(self.epsilon_bias.size(), out=self.epsilon_bias)
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(x, self.weight + self.sigma_weight * self.epsilon_weight, bias)


class Linear0(nn.Linear):
    def reset_parameters(self):
        nn.init.constant_(self.weight, 0.)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0.)


__all__ = ["Aggregation", "Dueling", "Flatten", "NoisyLinear", "Linear0"]
