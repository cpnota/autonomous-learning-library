import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .state_value_approximation import StateValueApproximation

class TorchStateValue(StateValueApproximation):
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = (optimizer
                          if optimizer is not None
                          else optim.Adam(model.parameters()))

    def __call__(self, state):
        with torch.no_grad():
            return self.model(state)

    def update(self, error, state):
        self.optimizer.zero_grad()
        value = self.model(state)
        value.backward(-error.view(value.shape))
        self.optimizer.step()
