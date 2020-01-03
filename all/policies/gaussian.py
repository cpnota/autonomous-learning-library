from torch.distributions.independent import Independent
from torch.distributions.normal import Normal
from all.approximation import Approximation
from all.nn import ListNetwork


class GaussianPolicy(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            **kwargs
    ):
        super().__init__(
            GaussianPolicyNetwork(model),
            optimizer,
            **kwargs
        )

class GaussianPolicyNetwork(ListNetwork):
    def __init__(self, model):
        super().__init__(model)

    def forward(self, state):
        outputs = super().forward(state)
        action_dim = outputs.shape[1] // 2
        means = outputs[:, 0:action_dim]
        logvars = outputs[:, action_dim:]
        std = logvars.mul(0.5).exp_()
        return Independent(Normal(means, std), 1)
