import torch
from all.core import State
from all.optim import Schedulable


class GeneralizedAdvantageBuffer(Schedulable):
    def __init__(
            self,
            v,
            features,
            n_steps,
            n_envs,
            discount_factor=1,
            lam=1
    ):
        self.v = v
        self.features = features
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.gamma = discount_factor
        self.lam = lam
        self._batch_size = self.n_steps * self.n_envs
        self._states = []
        self._actions = []
        self._rewards = []

    def __len__(self):
        return len(self._states) * self.n_envs

    def store(self, states, actions, rewards):
        if states is None:
            return
        if not self._states:
            self._states = [states]
            self._actions = [actions]
            self._rewards = [rewards]
        elif len(self._states) <= self.n_steps:
            self._states.append(states)
            self._actions.append(actions)
            self._rewards.append(rewards)
        else:
            raise Exception("Buffer length exceeded: " + str(self.n_steps))

    def advantages(self, states):
        if len(self) < self._batch_size:
            raise Exception("Not enough states received!")

        self._states.append(states)
        states = State.array(self._states[0:self.n_steps + 1])
        actions = torch.cat(self._actions[:self.n_steps], dim=0)
        rewards = torch.stack(self._rewards[:self.n_steps])
        _values = self.v.target(self.features.target(states))
        values = _values[0:self.n_steps]
        next_values = _values[1:]
        td_errors = rewards + self.gamma * next_values - values
        advantages = self._compute_advantages(td_errors)
        self._clear_buffers()
        return (
            states[0:-1].flatten(),
            actions,
            advantages.view(-1)
        )

    def _compute_advantages(self, td_errors):
        advantages = td_errors.clone()
        current_advantages = advantages[0] * 0

        # the final advantage is always 0
        advantages[-1] = current_advantages
        for i in range(self.n_steps):
            t = self.n_steps - 1 - i
            mask = self._states[t + 1].mask.float()
            current_advantages = td_errors[t] + self.gamma * self.lam * current_advantages * mask
            advantages[t] = current_advantages

        return advantages

    def _clear_buffers(self):
        self._states = []
        self._actions = []
        self._rewards = []
