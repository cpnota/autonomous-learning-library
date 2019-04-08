import torch
from torch.nn import functional
from all.layers import ListNetwork
from .abstract import Policy


class SoftmaxPolicy(Policy):
    def __init__(self, model, optimizer, actions, entropy_loss_scaling=0, clip_grad=0):
        self.model = ListNetwork(model, (actions,))
        self.optimizer = optimizer
        self.entropy_loss_scaling = entropy_loss_scaling
        self.clip_grad = clip_grad
        self._log_probs = []
        self._entropy = []

    def __call__(self, state, action=None, prob=None):
        scores = self.model(state)
        probs = functional.softmax(scores, dim=-1)
        distribution = torch.distributions.Categorical(probs)
        action = distribution.sample()
        self.cache(distribution, action)
        return action

    def eval(self, state):
        with torch.no_grad():
            scores = self.model(state)
            return functional.softmax(scores, dim=-1)

    def reinforce(self, errors):
        # shape the data properly
        errors = errors.view(-1)
        log_probs = torch.cat(self._log_probs).view(-1)
        entropy = torch.cat(self._entropy).view(-1)
        n = len(errors)

        # compute losses
        policy_loss = -torch.dot(log_probs, errors) / n
        entropy_loss = -entropy.mean()
        loss = policy_loss + self.entropy_loss_scaling * entropy_loss
        loss.backward()

        # take gradient steps
        self.optimizer.step()
        self.optimizer.zero_grad()

        # reset cache
        self._log_probs = []
        self._entropy = []

    def cache(self, distribution, action):
        self._log_probs.append(distribution.log_prob(action))
        self._entropy.append(distribution.entropy())
