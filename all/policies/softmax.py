import torch
from torch.nn import functional, utils
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

    def reinforce(self, errors, retain_graph=False):
        # shape the data properly
        errors = errors.view(-1)
        log_probs = torch.cat(self._log_probs)
        entropy = torch.cat(self._entropy)

        # adjust for batch size, fix cache
        # TODO make sure we don't cause memory leaks
        batch_size = len(errors)
        self._log_probs = [log_probs[batch_size:]]
        log_probs = log_probs[:batch_size]
        self._entropy = [entropy[batch_size:]]
        entropy = entropy[:batch_size]

        # compute losses
        policy_loss = -torch.dot(log_probs, errors) / batch_size
        entropy_loss = -entropy.mean()
        loss = policy_loss + self.entropy_loss_scaling * entropy_loss
        retain = retain_graph or len(log_probs) > 0
        loss.backward(retain_graph=retain)

        # take gradient steps
        if self.clip_grad != 0:
            utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def cache(self, distribution, action):
        self._log_probs.append(distribution.log_prob(action))
        self._entropy.append(distribution.entropy())
