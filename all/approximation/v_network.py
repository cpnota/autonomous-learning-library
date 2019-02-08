import torch
from torch import optim
from .v_function import ValueFunction

class ValueNetwork(ValueFunction):
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = (optimizer
                          if optimizer is not None
                          else optim.Adam(model.parameters()))
        self.cache = None

    def __call__(self, states):
        result = self._eval(states)
        self.cache = result
        return result.detach()

    def eval(self, states):
        with torch.no_grad():
            return self._eval(states)

    def reinforce(self, errors):
        self.cache.backward(-errors)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _eval(self, states):
        if isinstance(states, list):
            non_terminal_states = [
                state for state in states if state is not None]
            non_terminal_indexes = [
                i for i, state in enumerate(states) if state is not None]
            values = self.model(
                torch.cat(non_terminal_states).float())
            result = torch.zeros((len(states), values.shape[1]))
            result[non_terminal_indexes] = values
            return result.squeeze(1)
        if states is None:
            return torch.zeros(1)
        return self.model(states.float()).squeeze(1)
