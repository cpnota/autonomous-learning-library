import torch
from torch import optim
from torch.nn import functional
from .abstract import Policy


class SoftmaxPolicy(Policy):
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = (optimizer
                          if optimizer is not None
                          else optim.Adam(model.parameters()))
        self._cache = []

    def __call__(self, state, action=None, prob=None):
        scores = self.model(state.float())
        probs = functional.softmax(scores, dim=-1)
        distribution = torch.distributions.Categorical(probs)
        action = distribution.sample()
        self.cache(-distribution.log_prob(action))
        return action

    def eval(self, state):
        with torch.no_grad():
            scores = self.model(state.float())
            return functional.softmax(scores, dim=-1)

    def reinforce(self, errors):
        log_probs = torch.cat(self._cache)
        steps = log_probs.shape[0]
        loss = torch.bmm(log_probs.view(steps, 1, -1), errors.view(steps, -1, 1)).mean()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self._cache = []

    def cache(self, log_prob):
        self._cache.append(log_prob)
