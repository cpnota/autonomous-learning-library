import torch
from all.nn import ListNetwork, utils
from all.experiments import DummyWriter
from .policy import Policy


class DeterministicPolicy(Policy):
    def __init__(
            self,
            model,
            optimizer,
            space,
            noise,
            name='policy',
            # entropy_loss_scaling=0,
            clip_grad=0,
            writer=DummyWriter()
    ):
        self.model = ListNetwork(model, (space.shape[0],))
        self.optimizer = optimizer
        self.name = name
        self.device = next(model.parameters()).device
        self.noise = torch.distributions.normal.Normal(0, noise)
        self._low = torch.tensor(space.low, device=self.device)
        self._high = torch.tensor(space.high, device=self.device)
        # self._entropy_loss_scaling = entropy_loss_scaling
        self._clip_grad = clip_grad
        self._log_probs = []
        self._entropy = []
        self._writer = writer

    def __call__(self, state, action=None, prob=None):
        outputs = self.model(state).detach()
        outputs = outputs + self.noise.sample(outputs.shape).to(self.device)
        outputs = torch.min(outputs, self._high)
        outputs = torch.max(outputs, self._low)
        return outputs

    def greedy(self, state):
        return self.model(state)

    def eval(self, state):
        with torch.no_grad():
            return self.greedy(state)

    def reinforce(self):
        # Deterministic policies are trained through backprogation,
        # so reinforce() does not take any parameters.
        if self._clip_grad != 0:
            utils.clip_grad_norm_(self.model.parameters(), self._clip_grad)
        self.optimizer.step()
        self.optimizer.zero_grad()
