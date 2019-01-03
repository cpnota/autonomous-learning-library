import torch
from torch import optim
from .abstract import StateValue

class ContinuousStateValue(StateValue):
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = (optimizer
                          if optimizer is not None
                          else optim.Adam(model.parameters()))

    def __call__(self, state):
        if state is None:
            return 0

        with torch.no_grad():
            return self.model(state)

    def update(self, error, state):
        self.optimizer.zero_grad()
        value = self.model(state)
        value.backward(-error.view(value.shape))
        self.optimizer.step()
