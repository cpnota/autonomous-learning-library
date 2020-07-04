import torch
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
from all.approximation import Approximation
from all.nn import RLNetwork


class GaussianPolicy(Approximation):
    '''
    A Gaussian stochastic policy.

    This policy will choose actions from a distribution represented by a spherical Gaussian.
    The first n outputs the model will be squashed to [-1, 1] through a tanh function, and then
    scaled to the given action_space, and the remaining n outputs will define the amount of noise added.

    Args:
        model (torch.nn.Module): A Pytorch module representing the policy network.
            The input shape should be the same as the shape of the state (or feature) space,
            and the output shape should be double the size of the the action space.
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
        means = self._squash(outputs[:, 0:action_dim])

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
