import copy
import torch
from .abstract import TargetNetwork


class PolyakTarget(TargetNetwork):
    '''TargetNetwork that updates using polyak averaging'''

    def __init__(self, rate):
        self._source = None
        self._target = None
        self._rate = rate

    def __call__(self, *inputs):
        with torch.no_grad():
            return self._target(*inputs)

    def init(self, model):
        self._source = model
        self._target = copy.deepcopy(model)

    def update(self):
        for target_param, source_param in zip(self._target.parameters(), self._source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self._rate) + source_param.data * self._rate)
