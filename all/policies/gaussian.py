import torch
from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
from all.approximation import Approximation
from all.nn import RLNetwork


class GaussianPolicy(Approximation):
    '''
    A Gaussian stochastic policy.

    This policy will choose actions from a distribution represented by a spherical Gaussian.
    The first n outputs of the model are the mean of the distribution and the last n outputs are the log variance.
    The output will be centered and scaled to the size of the given space, but the output will not be clipped.
    For example, for an output range of [-1, 1], the center is 0 and the scale is 1.

    Args:
        model (torch.nn.Module): A Pytorch module representing the policy network.
            The input shape should be the same as the shape of the state (or feature) space,
            and the output shape should be double the size of the the action space.
            The first n outputs will be the unscaled mean of the action for each dimension,
            and the last n outputs will be the logarithm of the variance.
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
        action_dim = outputs.shape[-1] // 2
        means = outputs[..., 0:action_dim]
        logvars = outputs[..., action_dim:]
        std = (0.5 * logvars).exp_()
        return Independent(Normal(means + self._center, std * self._scale), 1)

    def to(self, device):
        self._center = self._center.to(device)
        self._scale = self._scale.to(device)
        return super().to(device)
