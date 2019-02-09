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

    def __call__(self, states, actions=None):
        result = self._eval(states, actions, self.model)
        self.cache = result
        return result.detach()

    def eval(self, states, actions=None):
        with torch.no_grad():
            return self._eval(states, actions, self.target_model)

    def reinforce(self, errors):
        self.cache.backward(-errors)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.updates += 1
        if self.should_update_target():
            self.target_model.load_state_dict(self.model.state_dict())

    def _eval(self, states, actions, model):
        values = self._eval_states(states, model)
        return (
            values if actions is None
            else values.transpose(0, 1)[actions]
        )

    def _eval_states(self, states, model):
        if isinstance(states, list):
            return self._eval_list(states, model)
        if states is None:
            return torch.zeros(1)
        return model(states.float())

    def _eval_list(self, states, model):
        non_terminal_states = [state for state in states if state is not None]
        non_terminal_indexes = [i for i, state in enumerate(states) if state is not None]
        non_terminal_values = model(torch.cat(non_terminal_states).float())
        result = torch.zeros((len(states), non_terminal_values.shape[1]))
        result[non_terminal_indexes] = non_terminal_values
        return result

    def should_update_target(self):
        return (
            (self.target_update_frequency is not None)
            and (self.updates % self.target_update_frequency == 0)
        )
