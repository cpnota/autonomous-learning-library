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

    def __call__(self, state, action=None, prob=None):
        outputs = self.model(state)
        distribution = self.distribution(outputs)
        if action is None:
            action = distribution.sample()
            self._enqueue(distribution, action)
            return action
        self._enqueue(distribution, action)
        return distribution.log_prob(action)

    def eval(self, state, action=None):
        with torch.no_grad():
            outputs = self.model(state)
            distribution = self.distribution(outputs)
            if action is None:
                action = distribution.sample()
                return action
            return distribution.log_prob(action)

    def reinforce(self, loss, retain_graph=False):
        if callable(loss):
            log_probs, entropy = self._dequeue_all()
            policy_loss = loss(log_probs)
        else:
            # shape the data properly
            errors = loss.view(-1)
            batch_size = len(errors)
            log_probs, entropy = self._dequeue(batch_size)
            policy_loss = (-log_probs.transpose(0, -1) * errors).mean()
        if log_probs.requires_grad:
            # compute losses
            entropy_loss = -entropy.mean()
            loss = policy_loss + self._entropy_loss_scaling * entropy_loss
            self._writer.add_loss(self._name, loss)
            self._writer.add_loss(self._name + '/pg', policy_loss)
            self._writer.add_loss(self._name + '/entropy', entropy_loss)
            loss.backward(retain_graph=retain_graph)
            # take gradient step
            self.step()

    def _enqueue(self, distribution, action):
        self._log_probs.append(distribution.log_prob(action))
        self._entropy.append(distribution.entropy())

    def _dequeue(self, batch_size):
        i = 0
        items = 0
        while items < batch_size and i < len(self._log_probs):
            items += len(self._log_probs[i])
            i += 1
        if items != batch_size:
            raise ValueError("Incompatible batch size: " + str(batch_size))

        log_probs = torch.cat(self._log_probs[:i])
        self._log_probs = self._log_probs[i:]
        entropy = torch.cat(self._entropy[:i])
        self._entropy = self._entropy[i:]

        return log_probs, entropy

    def _dequeue_all(self):
        log_probs = torch.cat(self._log_probs)
        self._log_probs = []
        entropy = torch.cat(self._entropy)
        self._entropy = []
        return log_probs, entropy
