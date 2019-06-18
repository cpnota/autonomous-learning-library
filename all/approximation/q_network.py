import torch
from torch import nn
from torch.nn.functional import mse_loss
from all.layers import ListNetwork
from .approximation import Approximation

def td_loss(loss):
    def _loss(estimates, errors):
        return loss(estimates, errors + estimates.detach())
    return _loss

class QModule(nn.Module):
    def __init__(self, model, num_actions):
        super().__init__()
        self.device = next(model.parameters()).device
        self.model = ListNetwork(model, (num_actions,))

    def forward(self, states, actions=None):
        values = self.model(states)
        if actions is None:
            return values
        if isinstance(actions, list):
            actions = torch.tensor(actions, device=self.device)
        return values.gather(1, actions.view(-1, 1)).squeeze(1)

class QNetwork(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            num_actions,
            loss=mse_loss,
            name='q',
            **kwargs
    ):
        model = QModule(model, num_actions)
        loss = td_loss(loss)
        super().__init__(
            model,
            optimizer,
            loss=loss,
            name=name,
            **kwargs
        )
