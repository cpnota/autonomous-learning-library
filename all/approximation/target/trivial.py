import torch
from .abstract import TargetNetwork

class TrivialTarget(TargetNetwork):
    def __init__(self):
        self._model = None

    def __call__(self, *inputs):
        with torch.no_grad():
            training = self._model.training
            result = self._model(*inputs)
            self._model.train(training)
            return result

    def init(self, model):
        self._model = model

    def update(self):
        pass
