import torch
from torch import optim
from torch.nn import functional
from .abstract import Policy


class SoftmaxPolicy(Policy):
    def __init__(self, model, optimizer=optim.Adam):
        self.model = model
        self.optimizer = optimizer(self.model.parameters())
        self._cache = torch.tensor([])

    def __call__(self, state, action=None, prob=None):
        scores = self.model(state)
        probs = functional.softmax(scores, dim=-1)
        distribution = torch.distributions.Categorical(probs)
        action = distribution.sample()
        self.cache(-distribution.log_prob(action))
        return action


    def reinforce(self, errors):
        print(self._cache.shape)
        print(errors.shape)
        loss = self._cache.dot(errors)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self._cache = torch.tensor([])

    def cache(self, log_prob):
        self._cache = torch.cat((self._cache, log_prob))
