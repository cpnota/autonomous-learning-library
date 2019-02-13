import torch
from torch import optim
from torch.nn.functional import smooth_l1_loss
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

    def reinforce(self, td_errors):
        targets = td_errors + self.cache.detach()
        loss = smooth_l1_loss(self.cache, targets)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _eval(self, states):
        if isinstance(states, list):
            return self._eval_list(states)
        if states is None:
            return torch.zeros(1)
        return self.model(states.float()).squeeze(1)

    def _eval_list(self, states):
        (non_terminal_states, non_terminal_indexes) = get_non_terminal_states(states)
        non_terminal_values = self.model(torch.cat(non_terminal_states).float())
        result = torch.zeros((len(states), non_terminal_values.shape[1]))
        result[non_terminal_indexes] = non_terminal_values
        return result.squeeze(1)

def get_non_terminal_states(states):
    non_terminal_states = [state for state in states if state is not None]
    non_terminal_indexes = [i for i, state in enumerate(states) if state is not None]
    return (non_terminal_states, non_terminal_indexes)
