import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .action_value_approximation import ActionValueApproximation

class TabularActionValue(ActionValueApproximation):
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = (optimizer
                          if optimizer is not None
                          else optim.Adam(model.parameters()))

    def __call__(self, state, action=None):
        if state is None:
            return 0

        with torch.no_grad():
            values = self.model(state)
            if action is None:
                return values
            else:
                return values[action]

    def update(self, error, state, action):
        self.optimizer.zero_grad()
        value = self.model(state)[action]
        value.backward(-error.view(value.shape))
        self.optimizer.step()
