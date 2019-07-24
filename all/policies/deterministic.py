import torch
from all.approximation import Approximation
from all.nn import ListNetwork


class DeterministicPolicy(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            space,
            noise,
            name='policy',
            **kwargs
    ):
        model = ListNetwork(model)
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )
        self.noise = torch.distributions.normal.Normal(0, noise)
        self._low = torch.tensor(space.low, device=self.device)
        self._high = torch.tensor(space.high, device=self.device)
        self._log_probs = []
        self._entropy = []

    def __call__(self, state, action=None, prob=None):
        outputs = self.model(state).detach()
        outputs = outputs + self.noise.sample(outputs.shape).to(self.device)
        outputs = torch.min(outputs, self._high)
        outputs = torch.max(outputs, self._low)
        return outputs

    def greedy(self, state):
        return self.model(state)

    def eval(self, state):
        return self._target(state)

    def reinforce(self, _):
        raise NotImplementedError(
            'Deterministic policies are trainted through backpropagation.' +
            'Call backward() on a loss derived from the action' +
            'and then call policy.step()'
        )
