from all.memory import NStepBuffer
from .abstract import Agent

class A2C(Agent):
    def __init__(self, v, policy, n_steps=1, batch_size=128, discount_factor=1):
        self.v = v
        self.policy = policy
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.discount_factor = discount_factor
        self._buffer = self._make_buffer()

    def act(self, states, rewards, info=None):
        self._buffer.store(states, rewards)
        if self._buffer.is_full():
            self._train()
        return self.policy(states)

    def _train(self):
        states, next_states, returns = self._buffer.sample(-1)
        td_errors = (
            returns
            + (self.discount_factor ** self.n_steps) * self.v.eval(next_states)
            - self.v(states)
        )
        self.v.reinforce(td_errors)
        self.policy.reinforce(td_errors)

    def _make_buffer(self):
        return NStepBuffer(self.n_steps, self.batch_size, discount_factor=self.discount_factor)
