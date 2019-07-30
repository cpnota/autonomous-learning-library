import torch
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
            lam=0.95
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
        self._epsilon = epsilon
        self._epochs = epochs
        self._batch_size = n_envs * n_steps
        self._minibatches = minibatches
        self._buffer = self._make_buffer()
        self._features = []

    def act(self, states, rewards):
        self._train(states)
        actions = self.policy.eval(self.features.eval(states))
        self._buffer.store(states, actions, rewards)
        return actions

    def _train(self, _states):
        if len(self._buffer) >= self._batch_size:
            states, actions, advantages = self._buffer.advantages(_states)
            with torch.no_grad():
                features = self.features.eval(states)
                pi_0 = self.policy.eval(features, actions)
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
        features = self.features(states)
        self.policy(features, actions)
        self.policy.reinforce(self._compute_policy_loss(pi_0, advantages))
        self.v.reinforce(targets - self.v(features))
        self.features.reinforce()

    def _compute_targets(self, returns, next_states, lengths):
        return (
            returns +
            (self.discount_factor ** lengths)
            * self.v.eval(self.features.eval(next_states))
        )

    def _compute_policy_loss(self, pi_0, advantages):
        def _policy_loss(pi_i):
            ratios = torch.exp(pi_i - pi_0)
            surr1 = ratios * advantages
            epsilon = self._epsilon
            surr2 = torch.clamp(ratios, 1.0 - epsilon, 1.0 + epsilon) * advantages
            return -torch.min(surr1, surr2).mean()
        return _policy_loss

    def _make_buffer(self):
        return GeneralizedAdvantageBuffer(
            self.v,
            self.features,
            self.n_steps,
            self.n_envs,
            discount_factor=self.discount_factor,
            lam=self.lam
        )
 