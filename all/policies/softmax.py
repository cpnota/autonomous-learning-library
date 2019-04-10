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
        batch_size = len(errors)
        log_probs, entropy = self.decache(batch_size)

        # compute losses
        policy_loss = -torch.dot(log_probs, errors) / batch_size
        entropy_loss = -entropy.mean()
        loss = policy_loss + self.entropy_loss_scaling * entropy_loss
        loss.backward(retain_graph=retain_graph)

        # take gradient steps
        if self.clip_grad != 0:
            utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def cache(self, distribution, action):
        self._log_probs.append(distribution.log_prob(action))
        self._entropy.append(distribution.entropy())

    def decache(self, batch_size):
        i = 0
        items = 0
        while items < batch_size:
            items += len(self._log_probs[i])
            i += 1
        if items != batch_size:
            raise ValueError("Incompatible batch size: " + str(batch_size))
        
        log_probs = torch.cat(self._log_probs[:i])
        self._log_probs = self._log_probs[i:]
        entropy = torch.cat(self._entropy[:i])
        self._entropy = self._entropy[i:]

        return log_probs, entropy