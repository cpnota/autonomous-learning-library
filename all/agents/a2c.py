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
        self._states = None
        self._actions = None
        self._batch_size = n_envs * n_steps
        self._buffer = self._make_buffer()
        self._features = []

    def act(self, states, rewards):
        self._store_transitions(rewards)
        self._train(states)
        self._states = states
        self._actions = self.policy.eval(self.features.eval(states))
        return self._actions

    def _store_transitions(self, rewards):
        if self._states:
            self._buffer.store(self._states, self._actions, rewards)

    def _train(self, states):
        if len(self._buffer) >= self._batch_size:
            states, actions, advantages = self._buffer.advantages(states)
            # forward pass
            features = self.features(states)
            values = self.v(features)
            log_pis = self.policy(features, actions)
            # backward pass
            self.v.reinforce(values, values.detach() + advantages)
            self.policy.reinforce(log_pis, advantages)
            self.features.reinforce()

    def _make_buffer(self):
        return NStepAdvantageBuffer(
            self.v,
            self.features,
            self.n_steps,
            self.n_envs,
            discount_factor=self.discount_factor
        )
 