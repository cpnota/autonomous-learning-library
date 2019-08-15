import torch
from torch.nn import functional as F
from all import nn
from .approximation import Approximation


class QDist(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            n_actions,
            n_atoms,
            v_min,
            v_max,
            name="q_dist",
            **kwargs
    ):
        model = QDistModule(model, n_actions, n_atoms)
        device = next(model.parameters()).device
        self.n_actions = n_actions
        self.atoms = torch.linspace(v_min, v_max, steps=n_atoms).to(device)
        super().__init__(model, optimizer, loss=cross_entropy_loss, name=name, **kwargs)


class QDistModule(nn.Module):
    def __init__(self, model, n_actions, n_atoms):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.device = next(model.parameters()).device
        self.terminal = torch.zeros((n_atoms)).to(self.device)
        self.terminal[(n_atoms // 2)] = 1.0
        self.model = nn.ListNetwork(model)
        self.count = 0

    def forward(self, states, actions=None):
        values = self.model(states).view((len(states), self.n_actions, self.n_atoms))
        values = F.softmax(values, dim=2)
        # trick to convert to terminal without manually looping
        values = (values - self.terminal) * states.mask.view(
            (-1, 1, 1)
        ).float() + self.terminal
        if actions is None:
            return values
        if isinstance(actions, list):
            actions = torch.stack(actions)
        return values[torch.arange(len(states)), actions]


def cross_entropy_loss(dist, target_dist):
    log_dist = torch.log(dist)
    loss_v = -log_dist * target_dist
    return loss_v.sum(dim=-1).mean()
