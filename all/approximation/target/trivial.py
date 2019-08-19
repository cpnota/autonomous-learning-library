import torch
from .abstract import TargetNetwork

class TrivialTarget(TargetNetwork):
    def __init__(self):
        self._model = None

    def __call__(self, *inputs):
        with torch.no_grad():
            training = self._model.training
            self._model.training = False
            out = self._model(*inputs)
            self._model.training = training
            return out

    def init(self, model):
        self._model = model

    def update(self):
        pass
