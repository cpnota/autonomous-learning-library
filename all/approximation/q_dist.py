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
        device = next(model.parameters()).device
        self.n_actions = n_actions
        self.atoms = torch.linspace(v_min, v_max, steps=n_atoms).to(device)
        model = QDistModule(model, n_actions, self.atoms)
        super().__init__(model, optimizer, name=name, **kwargs)

    def project(self, dist, support):
        target_dist = dist * 0
        atoms = self.atoms
        v_min = atoms[0]
        v_max = atoms[-1]
        delta_z = atoms[1] - atoms[0]
        batch_size = len(dist)
        n_atoms = len(atoms)
        # vectorized implementation of Algorithm 1
        tz_j = support.clamp(v_min, v_max)
        bj = (tz_j - v_min) / delta_z
        l = bj.floor().clamp(0, len(atoms) - 1)
        u = bj.ceil().clamp(0, len(atoms) - 1)
        # This part is a little tricky:
        # We have to flatten the matrix first and use index_add.
        # This approach is taken from Curt Park (under the MIT license):
        # https://github.com/Curt-Park/rainbow-is-all-you-need/blob/master/08.rainbow.ipynb
        offset = (
            torch.linspace(0, (batch_size - 1) * n_atoms, batch_size)
            .long()
            .unsqueeze(1)
            .expand(batch_size, n_atoms)
            .to(self.device)
        )
        target_dist.view(-1).index_add_(
            0, (l.long() + offset).view(-1), (dist * (u - bj)).view(-1)
        )
        target_dist.view(-1).index_add_(
            0, (u.long() + offset).view(-1), (dist * (bj - l)).view(-1)
        )
        return target_dist


class QDistModule(torch.nn.Module):
    def __init__(self, model, n_actions, atoms):
        super().__init__()
        self.atoms = atoms
        self.n_actions = n_actions
        self.n_atoms = len(atoms)
        self.device = next(model.parameters()).device
        self.terminal = torch.zeros((self.n_atoms)).to(self.device)
        self.terminal[(self.n_atoms // 2)] = 1.0
        self.model = nn.RLNetwork(model)
        self.count = 0

    def forward(self, states, actions=None):
        values = self.model(states).view((len(states), self.n_actions, self.n_atoms))
        values = F.softmax(values, dim=2)
        mask = states.mask

        # trick to convert to terminal without manually looping
        if torch.is_tensor(mask):
            values = (values - self.terminal) * states.mask.view(
                (-1, 1, 1)
            ).float() + self.terminal
        else:
            values = (values - self.terminal) * mask + self.terminal

        if actions is None:
            return values
        if isinstance(actions, list):
            actions = torch.cat(actions)
        return values[torch.arange(len(states)), actions]

    def to(self, device):
        self.device = device
        self.atoms = self.atoms.to(device)
        self.terminal = self.terminal.to(device)
        return super().to(device)
