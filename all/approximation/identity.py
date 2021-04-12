import torch
from torch import nn
from .approximation import Approximation


class Identity(Approximation):
    '''
    An Approximation that represents the identity function.

    Because the model has no parameters, reinforce and step do nothing.
    '''

    def __init__(self, device, name='identity', **kwargs):
        super().__init__(nn.Identity(), None, device=device, name=name, **kwargs)

    def reinforce(self):
        return self

    def step(self):
        return self
