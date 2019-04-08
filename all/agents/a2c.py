from all.memory import NStepBuffer
from .abstract import Agent

class A2C(Agent):
    def __init__(self, features, v, policy, n_steps=1, batch_size=128, discount_factor=1):
        self.features = features
        self.v = v
        self.policy = policy
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self._buffer = self._make_buffer()

    def act(self, states, rewards, info=None):
        features = self.features(states)
        self._buffer.store(features, rewards)
        if self._buffer.is_full():
            self._train()
        return self.policy(features)

    def _train(self):
        features, next_features, returns = self._buffer.sample(-1)
        td_errors = (
            returns
            + (self.discount_factor ** self.n_steps) * self.v.eval(next_features)
            - self.v(features)
        )
        self.v.reinforce(td_errors, retain_graph=True)
        self.policy.reinforce(td_errors)
        self.features.reinforce()

    def _make_buffer(self):
        return NStepBuffer(self.n_steps, self.batch_size, discount_factor=self.discount_factor)
