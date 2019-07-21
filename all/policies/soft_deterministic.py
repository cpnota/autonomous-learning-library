import torch
from all.approximation import TrivialTarget
from all.nn import ListNetwork, utils
from all.experiments import DummyWriter
from .policy import Policy


class SoftDeterministicPolicy(Policy):
    def __init__(
            self,
            model,
            optimizer,
            space,
            name="policy",
            target=None,
            clip_grad=0,
            writer=DummyWriter(),
    ):
        self.model = ListNetwork(model)
        self.optimizer = optimizer
        self.name = name
        self.device = next(model.parameters()).device
        self._action_dim = space.shape[0]
        self._target = target or TrivialTarget()
        self._target.init(self.model)
        self._clip_grad = clip_grad
        self._entropy = []
        self._writer = writer
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
        if log_prob:
            return self._squash(raw_actions), distribution.log_prob(raw_actions).sum(1, keepdim=True)
        return self._squash(raw_actions)

    def greedy(self, state):
        return self._squash(self.model(state)[:, 0:self._action_dim])

    def eval(self, state):
        return self._squash(self._target(state)[:, 0:self._action_dim])

    def reinforce(self, _):
        raise NotImplementedError(
            "Deterministic policies are trainted through backpropagation."
            + "Call backward() on a loss derived from the action"
            + "and then call policy.step()"
        )

    def step(self):
        if self._clip_grad != 0:
            utils.clip_grad_norm_(self.model.parameters(), self._clip_grad)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self._target.update()

    def _distribution(self, outputs):
        means = outputs[:, 0 : self._action_dim]
        logvars = outputs[:, self._action_dim :]
        std = logvars.mul(0.5).exp_()
        self._last_dist = torch.distributions.normal.Normal(means, std)
        return self._last_dist

    def _squash(self, x):
        return torch.tanh(x) * self._tanh_scale + self._tanh_mean
