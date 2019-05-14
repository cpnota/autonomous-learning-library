from all.memory import NStepBatchBuffer
from .abstract import Agent


class A2C(Agent):
    def __init__(
            self,
            features,
            v,
            policy,
            n_envs=1,
            n_steps=4,
            discount_factor=0.99
    ):
        self.features = features
        self.v = v
        self.policy = policy
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.discount_factor = discount_factor
        self._batch_size = n_envs * n_steps
        self._buffer = self._make_buffer()

    def act(self, states, rewards, info=None):
        while len(self._buffer) >= self._batch_size:
            self._train()
        actions = self.policy(self.features(states))
        self._buffer.store(states, rewards)
        return actions

    def _train(self):
        states, next_states, returns, rollout_lengths = self._buffer.sample(self._batch_size)
        features = self.features(states)
        next_features = self.features(next_states)
        td_errors = (
            returns
            + (self.discount_factor ** rollout_lengths)
            * self.v.eval(next_features)
            - self.v(features)
        )
        self.v.reinforce(td_errors, retain_graph=True)
        self.policy.reinforce(td_errors)
        self.features.reinforce()

    def _make_buffer(self):
        return NStepBatchBuffer(
            self.n_steps,
            self.n_envs,
            discount_factor=self.discount_factor
        )
