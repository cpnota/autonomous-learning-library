import torch
from all.nn import ListNetwork
from .stochastic import StochasticPolicy


class GaussianPolicy(StochasticPolicy):
    def __init__(
            self,
            model,
            optimizer,
            action_dim,
            **kwargs
    ):
        model = ListNetwork(model, (action_dim * 2,))
        optimizer = optimizer

        def distribution(outputs):
            means = outputs[:, 0:action_dim]
            logvars = outputs[:, action_dim:]
            std = logvars.mul(0.5).exp_()
            return torch.distributions.normal.Normal(means, std)

        super().__init__(
            model,
            optimizer,
            distribution,
            **kwargs
        )
