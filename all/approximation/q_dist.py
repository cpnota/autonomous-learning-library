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
            name='q_dist',
            **kwargs
    ):
        model = QDistModule(model, n_actions, n_atoms)
        self.n_actions = n_actions
        self.atoms = torch.linspace(v_min, v_max, steps=n_atoms)
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )

class QDistModule(nn.Module):
    def __init__(self, model, n_actions, n_atoms):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        self.zero_atom = n_atoms // 2
        self.device = next(model.parameters()).device
        self.model = nn.ListNetwork(model)

    def forward(self, states, actions=None):
        values = self.model(states).view((len(states), self.n_actions, self.n_atoms))
        values = F.softmax(values, dim=2)
        for i in range(len(states)):
            if states.mask[i] == 0:
                values[i] *= 0
                values[i,:,self.zero_atom] = 1
        if actions is None:
            return values
        if isinstance(actions, list):
            actions = torch.tensor(actions, device=self.device)
        return values[torch.arange(len(states)), actions]
