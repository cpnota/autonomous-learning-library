import torch
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
from all.approximation import Approximation
from all.nn import RLNetwork


class GaussianPolicy(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            space,
            name='policy',
            **kwargs
    ):
        super().__init__(
            GaussianPolicyNetwork(model, space),
            optimizer,
            name=name,
            **kwargs
        )

class GaussianPolicyNetwork(RLNetwork):
    def __init__(self, model, space):
        super().__init__(model)
        self._center = torch.tensor((space.high + space.low) / 2).to(self.device)
        self._scale = torch.tensor((space.high - space.low) / 2).to(self.device)

    def forward(self, state):
        outputs = super().forward(state)
        action_dim = outputs.shape[1] // 2
        means = self._squash(torch.tanh(outputs[:, 0:action_dim]))

        if not self.training:
            return means

        logvars = outputs[:, action_dim:] * self._scale
        std = logvars.exp_()
        return Independent(Normal(means, std), 1)

    def _squash(self, x):
        return torch.tanh(x) * self._scale + self._center

    def to(self, device):
        self._center = self._center.to(device)
        self._scale = self._scale.to(device)
        return super().to(device)
