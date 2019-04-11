import copy
import torch
from torch import optim
from torch.nn.functional import mse_loss
from all.experiments import DummyWriter
from all.layers import ListNetwork
from .q_function import QFunction


class QNetwork(QFunction):
    def __init__(
            self,
            model,
            optimizer,
            actions,
            loss=mse_loss,
            target_update_frequency=None,
            writer=DummyWriter()
    ):
        self.model = ListNetwork(model, (actions,))
        self.optimizer = (optimizer
                          if optimizer is not None
                          else optim.Adam(model.parameters()))
        self.loss = loss
        self.cache = None
        self.updates = 0
        self.target_update_frequency = target_update_frequency
        self.target_model = (
            copy.deepcopy(self.model)
            if target_update_frequency is not None
            else self.model
        )
        self.device = next(model.parameters()).device
        self.writer = writer

    def __call__(self, states, actions=None):
        result = self._eval(states, actions, self.model)
        if result.requires_grad:
            self.cache = result
        return result.detach()

    def eval(self, states, actions=None):
        with torch.no_grad():
            training = self.target_model.training
            result = self._eval(states, actions, self.target_model.eval())
            self.target_model.train(training)
            return result

    def reinforce(self, td_errors):
        targets = td_errors + self.cache.detach()
        loss = self.loss(self.cache, targets)
        self.writer.add_loss('q', loss)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.updates += 1
        if self.should_update_target():
            self.target_model.load_state_dict(self.model.state_dict())

    def _eval(self, states, actions, model):
        values = model(states)
        if actions is None:
            return values
        if isinstance(actions, list):
            actions = torch.tensor(actions, device=self.device)
        return values.gather(1, actions.view(-1, 1)).squeeze(1)

    def should_update_target(self):
        return (
            (self.target_update_frequency is not None)
            and (self.updates % self.target_update_frequency == 0)
        )
