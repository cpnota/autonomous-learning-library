import torch
from torch import optim
from .abstract import ActionValue

class TabularActionValue(ActionValue):
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = (optimizer
                          if optimizer is not None
                          else optim.Adam(model.parameters()))
        self.cache = None

    def __call__(self, state, action=None):
        if isinstance(state, list):
            state = torch.cat(state)
        value = self.model(state.float())
        result = value.transpose(0, 1)[action]
        self.cache = result
        return result

    def eval(self, states):
        with torch.no_grad():
            if isinstance(states, list):
                non_terminal_states = [state for state in states if state is not None]
                non_terminal_indexes = [i for i, state in enumerate(states) if state is not None]
                values = self.model(torch.cat(non_terminal_states).float())

                result = torch.zeros((len(states), values.shape[1]))
                result[non_terminal_indexes] = values
                return result

            return self.model(states.float())

    def update(self, error, state, action):
        self.optimizer.zero_grad()
        value = self.model(state.float()).transpose(0, 1)[action]
        value.backward(-error.view(value.shape))
        self.optimizer.step()

    def reinforce(self, errors):
        self.cache.backward(-errors)
        self.optimizer.step()
        self.optimizer.zero_grad()
