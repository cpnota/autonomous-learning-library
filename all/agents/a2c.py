from all.memory import NStepAdvantageBuffer
from ._agent import Agent


class A2C(Agent):
    def __init__(
            self,
            features,
            v,
            policy,
            n_envs=None,
            n_steps=4,
            discount_factor=0.99
    ):
        if n_envs is None:
            raise RuntimeError("Must specify n_envs.")
        self.features = features
        self.v = v
        self.policy = policy
        self.n_envs = n_envs
        self.n_steps = n_steps
        self.discount_factor = discount_factor
        self._batch_size = n_envs * n_steps
        self._buffer = self._make_buffer()
        self._features = []

    def act(self, states, rewards):
        self._train(states)
        actions = self.policy.eval(self.features.eval(states))
        self._buffer.store(states, actions, rewards)
        return actions

    def _train(self, states):
        if len(self._buffer) >= self._batch_size:
            states, actions, advantages = self._buffer.advantages(states)
            # forward pass
            features = self.features(states)
            self.v(features)
            self.policy(features, actions)
            # backward pass
            self.v.reinforce(advantages)
            self.policy.reinforce(advantages)
            self.features.reinforce()

    def _make_buffer(self):
        return NStepAdvantageBuffer(
            self.v,
            self.features,
            self.n_steps,
            self.n_envs,
            discount_factor=self.discount_factor
        )
 