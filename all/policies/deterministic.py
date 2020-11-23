import torch
from all.approximation import Approximation
from all.nn import RLNetwork


class DeterministicPolicy(Approximation):
    '''
    A DDPG-style deterministic policy.

    Args:
        model (torch.nn.Module): A Pytorch module representing the policy network.
            The input shape should be the same as the shape of the state space,
            and the output shape should be the same as the shape of the action space.
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
        model = DeterministicPolicyNetwork(model, space)
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )


class DeterministicPolicyNetwork(RLNetwork):
    def __init__(self, model, space):
        super().__init__(model)
        self._action_dim = space.shape[0]
        self._tanh_scale = torch.tensor((space.high - space.low) / 2).to(self.device)
        self._tanh_mean = torch.tensor((space.high + space.low) / 2).to(self.device)

    def forward(self, state):
        return self._squash(super().forward(state))

    def _squash(self, x):
        return torch.tanh(x) * self._tanh_scale + self._tanh_mean

    def to(self, device):
        self._tanh_mean = self._tanh_mean.to(device)
        self._tanh_scale = self._tanh_scale.to(device)
        return super().to(device)
