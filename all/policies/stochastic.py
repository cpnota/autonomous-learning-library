import torch
from all.approximation import Approximation


class StochasticPolicy(Approximation):
    def __init__(
            self,
            model,
            optimizer,
            distribution,
            name='policy',
            entropy_loss_scaling=0,
            **kwargs
    ):
        super().__init__(
            model,
            optimizer,
            name=name,
            **kwargs
        )
        self.distribution = distribution
        self._entropy_loss_scaling = entropy_loss_scaling
        self._log_probs = []
        self._entropy = []

    def __call__(self, state, action=None):
        outputs = self.model(state)
        distribution = self.distribution(outputs)
        if action is None:
            action = distribution.sample()
            log_prob = distribution.log_prob(action)
            log_prob.entropy = distribution.entropy()
            return action, log_prob
        log_prob = distribution.log_prob(action)
        log_prob.entropy = distribution.entropy()
        return log_prob

    def eval(self, state, action=None):
        with torch.no_grad():
            outputs = self.model(state)
            distribution = self.distribution(outputs)
            if action is None:
                return distribution.sample()
            return distribution.log_prob(action)

    def reinforce(self, log_probs, advantages, retain_graph=False):
        loss = self.loss(log_probs, advantages)
        loss.backward(retain_graph=retain_graph)
        self.step()

    def loss(self, log_probs, advantages):
        policy_loss = (-log_probs.transpose(0, -1) * advantages.view(-1)).mean()
        entropy_loss = -log_probs.entropy.mean()
        loss = policy_loss + self._entropy_loss_scaling * entropy_loss
        self._writer.add_loss(self._name, loss)
        self._writer.add_loss(self._name + '/pg', policy_loss)
        self._writer.add_loss(self._name + '/entropy', entropy_loss)
        return loss
