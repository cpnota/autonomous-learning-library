import torch
from all.approximation import Approximation
from all.nn import RLNetwork

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
        self._inner_model = model
        super().__init__(model, optimizer, name=name, **kwargs)


class SoftDeterministicPolicyNetwork(RLNetwork):
    def __init__(self, model, space):
        super().__init__(model)
        self._action_dim = space.shape[0]
        self._tanh_scale = torch.tensor((space.high - space.low) / 2).to(self.device)
        self._tanh_mean = torch.tensor((space.high + space.low) / 2).to(self.device)

    def forward(self, state):
        outputs = super().forward(state)
        normal = self._normal(outputs)
        if self.training:
            action, log_prob = self._sample(normal)
            return action, log_prob
        return self._squash(normal.loc)

    def _normal(self, outputs):
        means = outputs[:, 0 : self._action_dim]
        logvars = outputs[:, self._action_dim:]
        std = logvars.mul(0.5).exp_()
        return torch.distributions.normal.Normal(means, std)

    def _sample(self, normal):
        raw = normal.rsample()
        action = self._squash(raw)
        log_prob = normal.log_prob(raw)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1)
        return action, log_prob

    def _squash(self, x):
        return torch.tanh(x) * self._tanh_scale + self._tanh_mean

    def to(self, device):
        self._tanh_mean = self._tanh_mean.to(device)
        self._tanh_scale = self._tanh_scale.to(device)
        return super().to(device)
