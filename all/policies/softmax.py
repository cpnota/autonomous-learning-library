import torch
from torch.nn import functional
from all.nn import RLNetwork
from all.approximation import Approximation


class SoftmaxPolicy(Approximation):
    '''
    A softmax (or Boltzmann) stochastic policy for discrete actions.

    Args:
        model (torch.nn.Module): A Pytorch module representing the policy network.
            The input shape should be the same as the shape of the state (or feature) space,
            and the output should be a vector the size of the action set.
        optimizer (torch.optim.Optimizer): A optimizer initialized with the
            model parameters, e.g. SGD, Adam, RMSprop, etc.
        kwargs (optional): Any other arguments accepted by all.approximation.Approximation
    '''

    def __init__(
            self,
            model,
            optimizer=None,
            name='policy',
            **kwargs
    ):
        model = SoftmaxPolicyNetwork(model)
        super().__init__(model, optimizer, name=name, **kwargs)


class SoftmaxPolicyNetwork(RLNetwork):
    def __init__(self, model):
        super().__init__(model)

    def forward(self, state):
        outputs = super().forward(state)
        probs = functional.softmax(outputs, dim=-1)
        return torch.distributions.Categorical(probs)
