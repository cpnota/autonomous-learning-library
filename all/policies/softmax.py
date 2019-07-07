import torch
from torch.nn import functional
from all.nn import ListNetwork
from .stochastic import StochasticPolicy


class SoftmaxPolicy(StochasticPolicy):
    def __init__(
            self,
            model,
            optimizer,
            actions,
            **kwargs
    ):
        model = ListNetwork(model, (actions,))

        def distribution(outputs):
            probs = functional.softmax(outputs, dim=-1)
            return torch.distributions.Categorical(probs)

        super().__init__(model, optimizer, distribution, **kwargs)
