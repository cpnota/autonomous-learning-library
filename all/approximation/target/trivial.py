import torch
from .abstract import TargetNetwork


class TrivialTarget(TargetNetwork):
    def __init__(self):
        self._model = None

    def __call__(self, *inputs):
        with torch.no_grad():
            return self._model(*inputs)

    def init(self, model):
        self._model = model

    def update(self):
        pass
