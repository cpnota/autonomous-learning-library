import torch
from all.approximation import Approximation
from all.nn import ListNetwork

class SoftDeterministicPolicy(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            space,
            name="policy",
            **kwargs
    ):
        model = SoftDeterministicPolicyNetwork(model, space)
        super().__init__(model, optimizer, name=name, **kwargs)


class SoftDeterministicPolicyNetwork(ListNetwork):
    def __init__(self, model, space):
        super().__init__(model)
        self._space = space
        self._action_dim = space.shape[0]

    def forward(self, state):
        outputs = super().forward(state)
        means = outputs[:, 0 : self._action_dim]
        logvars = outputs[:, self._action_dim:]
        std = logvars.mul(0.5).exp_()
        return SquashedNormal(means, std, self._space)


class SquashedNormal():
    '''A normal distribution squashed through a tanh and rescaled'''
    def __init__(self, loc, scale, space):
        self._normal = torch.distributions.normal.Normal(loc, scale)
        self._tanh_scale = torch.tensor((space.high - space.low) / 2, device=loc.device)
        self._tanh_mean = torch.tensor((space.high + space.low) / 2, device=loc.device)

    @property
    def mean(self):
        return self._squash(self._normal.mean)

    def sample(self, **kwargs):
        return self._squash(self._normal.sample(**kwargs))

    def rsample(self, **kwargs):
        raw = self._normal.rsample(**kwargs)
        action = self._squash(raw)
        log_prob = self._normal.log_prob(raw)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1)
        return action, log_prob

    def log_prob(self, action):
        # Squashing using tanh changes the probabilty densities.
        # See section C. of the Soft Actor-Critic appendix.
        unsquashed = self._unsquash(action)
        return self._normal.log_prob(unsquashed).sum(1) - torch.log(1 - torch.tanh(unsquashed) ** 2 + 1e-6).sum(1)

    def _squash(self, x):
        return torch.tanh(x) * self._tanh_scale + self._tanh_mean

    def _unsquash(self, x):
        return atanh((x - self._tanh_mean) / self._tanh_scale)

def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))
