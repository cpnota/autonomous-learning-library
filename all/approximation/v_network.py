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
        if isinstance(states, list):
            states = torch.cat(states)
        result = self.model(states.float())
        self.cache = result
        return result

    def eval(self, states):
        with torch.no_grad():
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
            return self.model(states.float()).squeeze(1)

    def reinforce(self, errors):
        self.cache.backward(-errors)
        self.optimizer.step()
        self.optimizer.zero_grad()
