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

    def __call__(self, state, action=None, log_prob=False):
        outputs = self.model(state)
        distribution = self.distribution(outputs)
        if action is None:
            action = distribution.sample()
            if log_prob:
                _log_prob = distribution.log_prob(action)
                _log_prob.entropy = distribution.entropy()
                return action, _log_prob
            return action
        _log_prob = distribution.log_prob(action)
        _log_prob.entropy = distribution.entropy()
        return _log_prob

    def eval(self, state, action=None, log_prob=False):
        with torch.no_grad():
            return self(state, action=action, log_prob=log_prob)

    def reinforce(self, log_probs, policy_loss, retain_graph=False):
        entropy_loss = -log_probs.entropy.mean()
        loss = policy_loss + self._entropy_loss_scaling * entropy_loss
        self._writer.add_loss(self._name, loss)
        self._writer.add_loss(self._name + '/pg', policy_loss)
        self._writer.add_loss(self._name + '/entropy', entropy_loss)
        loss.backward(retain_graph=retain_graph)
        self.step()
