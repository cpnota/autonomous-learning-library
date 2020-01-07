import torch
from torch.nn.functional import mse_loss
from all.logging import DummyWriter
from all.memory import GeneralizedAdvantageBuffer
from ._agent import Agent


class PPO(Agent):
    def __init__(
            self,
            features,
            v,
            policy,
            epsilon=0.2,
            epochs=4,
            minibatches=4,
            n_envs=None,
            n_steps=4,
            discount_factor=0.99,
            lam=0.95,
            entropy_loss_scaling=0.01,
            writer=DummyWriter()
    ):
        if n_envs is None:
            raise RuntimeError("Must specify n_envs.")
        self.features = features
        self.v = v
        self.policy = policy
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.discount_factor = discount_factor
        self.lam = lam
        self.entropy_loss_scaling = entropy_loss_scaling
        self.writer = writer
        self._states = None
        self._actions = None
        self._epsilon = epsilon
        self._epochs = epochs
        self._batch_size = n_envs * n_steps
        self._minibatches = minibatches
        self._buffer = self._make_buffer()
        self._features = []

    def act(self, states, rewards):
        self._store_transitions(rewards)
        self._train(states)
        self._states = states
        self._actions = self.policy.eval(self.features.eval(states)).sample()
        return self._actions

    def _store_transitions(self, rewards):
        if self._states:
            self._buffer.store(self._states, self._actions, rewards)

    def _train(self, _states):
        if len(self._buffer) >= self._batch_size:
            states, actions, advantages = self._buffer.advantages(_states)
            with torch.no_grad():
                features = self.features.eval(states)
                pi_0 = self.policy.eval(features).log_prob(actions)
                targets = self.v.eval(features) + advantages
            for _ in range(self._epochs):
                self._train_epoch(states, actions, pi_0, advantages, targets)

    def _train_epoch(self, states, actions, pi_0, advantages, targets):
        minibatch_size = int(self._batch_size / self._minibatches)
        indexes = torch.randperm(self._batch_size)
        for n in range(self._minibatches):
            first = n * minibatch_size
            last = first + minibatch_size
            i = indexes[first:last]
            self._train_minibatch(states[i], actions[i], pi_0[i], advantages[i], targets[i])

    def _train_minibatch(self, states, actions, pi_0, advantages, targets):
        # forward pass
        features = self.features(states)
        values = self.v(features)
        distribution = self.policy(features)
        pi_i = distribution.log_prob(actions)
        # compute losses
        value_loss = mse_loss(values, targets)
        policy_gradient_loss = self._policy_loss(pi_i, pi_0, advantages)
        entropy_loss = -distribution.entropy().mean()
        policy_loss = policy_gradient_loss + self.entropy_loss_scaling * entropy_loss
        # backward pass
        self.v.reinforce(value_loss)
        self.policy.reinforce(policy_loss)
        self.features.reinforce()
        # debugging
        self.writer.add_loss('policy_gradient', policy_gradient_loss.detach())
        self.writer.add_loss('entropy', -entropy_loss.detach())

    def _policy_loss(self, pi_i, pi_0, advantages):
        ratios = torch.exp(pi_i - pi_0)
        surr1 = ratios * advantages
        epsilon = self._epsilon
        surr2 = torch.clamp(ratios, 1.0 - epsilon, 1.0 + epsilon) * advantages
        return -torch.min(surr1, surr2).mean()

    def _make_buffer(self):
        return GeneralizedAdvantageBuffer(
            self.v,
            self.features,
            self.n_steps,
            self.n_envs,
            discount_factor=self.discount_factor,
            lam=self.lam
        )
 