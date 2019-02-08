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
        self._cache = torch.tensor([])

    def __call__(self, state, action=None, prob=None):
        scores = self.model(state)
        probs = functional.softmax(scores, dim=-1)
        distribution = torch.distributions.Categorical(probs)
        action = distribution.sample()
        self.cache(-distribution.log_prob(action))
        return action

    def reinforce(self, errors):
        loss = self._cache.dot(errors.detach())
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self._cache = torch.tensor([])

    def cache(self, log_prob):
        self._cache = torch.cat((self._cache, log_prob))
