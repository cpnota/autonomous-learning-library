import torch
from torch import optim
from torch.nn import functional
from .abstract import Policy


class SoftmaxPolicy(Policy):
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters())

    def __call__(self, state):
        with torch.no_grad():
            scores = self.model(state)
            probs = functional.softmax(scores, dim=-1)
            return torch.distributions.Categorical(probs).sample().item()

    def update(self, error, state, action):
        self.optimizer.zero_grad()
        scores = self.model(state)
        action = torch.tensor(action)
        log_probs = functional.log_softmax(
            scores, dim=-1).gather(-1, action.unsqueeze(-1))
        loss = -(log_probs * error).sum()
        loss.backward()
        self.optimizer.step()
