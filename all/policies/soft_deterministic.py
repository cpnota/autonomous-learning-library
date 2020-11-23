import torch
from all.approximation import Approximation
from all.nn import RLNetwork


class SoftDeterministicPolicy(Approximation):
    '''
    A "soft" deterministic policy compatible with soft actor-critic (SAC).

    Args:
        model (torch.nn.Module): A Pytorch module representing the policy network.
            The input shape should be the same as the shape of the state (or feature) space,
            and the output shape should be double the size of the the action space
            The first n outputs will be the unscaled mean of the action for each dimension,
            and the second n outputs will be the logarithm of the variance.
        optimizer (torch.optim.Optimizer): A optimizer initialized with the
            model parameters, e.g. SGD, Adam, RMSprop, etc.
        action_space (gym.spaces.Box): The Box representing the action space.
        kwargs (optional): Any other arguments accepted by all.approximation.Approximation
    '''

    def __init__(
            self,
            model,
            optimizer=None,
            space=None,
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
        means = outputs[..., 0:self._action_dim]
        logvars = outputs[..., self._action_dim:]
        std = logvars.mul(0.5).exp_()
        return torch.distributions.normal.Normal(means, std)

    def _sample(self, normal):
        raw = normal.rsample()
        log_prob = self._log_prob(normal, raw)
        return self._squash(raw), log_prob

    def _log_prob(self, normal, raw):
        '''
        Compute the log probability of a raw action after the action is squashed.
        Both inputs act on the raw underlying distribution.
        Because tanh_mean does not affect the density, we can ignore it.
        However, tanh_scale will affect the relative contribution of each component.'
        See Appendix C in the Soft Actor-Critic paper

        Args:
            normal (torch.distributions.normal.Normal): The "raw" normal distribution.
            raw (torch.Tensor): The "raw" action.

        Returns:
            torch.Tensor: The probability of the raw action, accounting for the affects of tanh.
        '''
        log_prob = normal.log_prob(raw)
        log_prob -= torch.log(1 - torch.tanh(raw).pow(2) + 1e-6)
        log_prob /= self._tanh_scale
        return log_prob.sum(-1)

    def _squash(self, x):
        return torch.tanh(x) * self._tanh_scale + self._tanh_mean

    def to(self, device):
        self._tanh_mean = self._tanh_mean.to(device)
        self._tanh_scale = self._tanh_scale.to(device)
        return super().to(device)
