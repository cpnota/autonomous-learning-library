import torch
from torch import optim
from torch.nn import functional
from .abstract import Policy


class SoftmaxPolicy(Policy):
    def __init__(self, model, optimizer=optim.Adam):
        self.model = model
        self.optimizer = optimizer(self.model.parameters())
        self._cache = torch.tensor([])

    def __call__(self, state):
        with torch.no_grad():
            scores = self.model(state)
            probs = functional.softmax(scores, dim=-1)
            return torch.distributions.Categorical(probs).sample().item()

    def execute(self, state):
        scores = self.model(state)
        probs = functional.softmax(scores, dim=-1)
        distribution = torch.distributions.Categorical(probs)
        action = distribution.sample()
        self.cache(-distribution.log_prob(action))
        return action

    def reinforce(self, errors):
        loss = self._cache.dot(errors)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self._cache = torch.tensor([])

    def update(self, error, state, action):
        self.optimizer.zero_grad()
        scores = self.model(state)
        action = torch.tensor(action)
        log_probs = functional.log_softmax(
            scores, dim=-1).gather(-1, action.unsqueeze(-1))
        loss = -(log_probs * error).sum()
        loss.backward()
        self.optimizer.step()

    def cache(self, log_prob):
        self._cache = torch.cat((self._cache, log_prob.unsqueeze(0)))
