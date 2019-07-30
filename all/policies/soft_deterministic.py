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
        model = ListNetwork(model)
        optimizer = optimizer
        name = name
        super().__init__(model, optimizer, name=name, **kwargs)
        self._action_dim = space.shape[0]
        self._entropy = []
        self._last_dist = None
        self._raw_actions = None
        # parameters for squashing to tanh
        self._tanh_scale = torch.tensor(
            (space.high - space.low) / 2, device=self.device
        )
        self._tanh_mean = torch.tensor((space.high + space.low) / 2, device=self.device)

    def __call__(self, state, action=None, log_prob=False):
        outputs = self.model(state)
        distribution = self._distribution(outputs)
        raw_actions = distribution.rsample()
        actions = self._squash(raw_actions)
        if log_prob:
            return (actions, self._log_prob(
                distribution,
                raw_actions,
                actions
            ))
        return actions

    def greedy(self, state):
        return self._squash(self.model(state)[:, 0:self._action_dim])

    def eval(self, state):
        with torch.no_grad():
            return self(state)

    def target(self, state):
        self._target(state)

    def reinforce(self, _):
        raise NotImplementedError(
            "Deterministic policies are trainted through backpropagation."
            + "Call backward() on a loss derived from the action"
            + "and then call policy.step()"
        )

    def _distribution(self, outputs):
        means = outputs[:, 0 : self._action_dim]
        logvars = outputs[:, self._action_dim :]
        std = logvars.mul(0.5).exp_()
        self._last_dist = torch.distributions.normal.Normal(means, std)
        return self._last_dist

    def _squash(self, x):
        return torch.tanh(x) * self._tanh_scale + self._tanh_mean

    def _log_prob(self, dist, raw, action):
        # Squashing using tanh changes the probabilty densities.
        # See section C. of the Soft Actor-Critic appendix.
        log_prob = dist.log_prob(raw)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        return log_prob.sum(1)
