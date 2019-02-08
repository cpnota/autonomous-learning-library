import copy
import torch
from torch import optim
from .q_function import QFunction


class QTabular(QFunction):
    def __init__(self, model, optimizer=None, target_update_frequency=None):
        self.model = model
        self.optimizer = (optimizer
                          if optimizer is not None
                          else optim.Adam(model.parameters()))
        self.cache = None
        self.updates = 0
        self.target_update_frequency = target_update_frequency
        self.target_model = (
            copy.deepcopy(self.model)
            if target_update_frequency is not None
            else self.model
        )

    def __call__(self, state, action=None):
        if isinstance(state, list):
            state = torch.cat(state)
        value = self.model(state.float())
        result = value.transpose(0, 1)[action] if action is not None else value
        self.cache = result
        return result

    def eval(self, states, actions=None):
        with torch.no_grad():
            if isinstance(states, list):
                non_terminal_states = [
                    state for state in states if state is not None]
                non_terminal_indexes = [
                    i for i, state in enumerate(states) if state is not None]
                values = self.target_model(
                    torch.cat(non_terminal_states).float())
                result = torch.zeros((len(states), values.shape[1]))
                result[non_terminal_indexes] = values
                return result
            if states is None:
                return torch.zeros(1)
            values = self.target_model(states.float())
            result = values.transpose(0, 1)[actions] if actions is not None else values
            return result


    def reinforce(self, errors):
        self.cache.backward(-errors)
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.updates += 1
        if (self.target_update_frequency is not None) and (self.updates % self.target_update_frequency == 0):
            self.target_model.load_state_dict(self.model.state_dict())
