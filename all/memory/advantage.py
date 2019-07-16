import torch
from all.environments import State

class NStepAdvantageBuffer:
    def __init__(self, v, features, n_steps, n_envs, discount_factor=1):
        self.v = v
        self.features = features
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.gamma = discount_factor
        self._states = []
        self._actions = []
        self._rewards = []

    def __len__(self):
        if not self._states:
            return 0
        return (len(self._states) - 1) * self.n_envs

    def store(self, states, actions, rewards):
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

    def sample(self, _):
        if len(self) < self.n_steps * self.n_envs:
            raise Exception("Not enough states received!")

        sample_n = self.n_envs * self.n_steps
        sample_states = [None] * sample_n
        sample_actions = [None] * sample_n
        sample_returns = [0] * sample_n
        sample_next_states = [None] * sample_n
        sample_lengths = [0] * sample_n

        # compute the N-step returns the slow way
        sample_returns = torch.zeros(
            (self.n_steps, self.n_envs),
            device=self._rewards[0].device
        )
        sample_lengths = torch.zeros(
            (self.n_steps, self.n_envs),
            device=self._rewards[0].device
        )
        current_returns = self._rewards[0] * 0
        current_lengths = current_returns.clone()
        for i in range(self.n_steps):
            t = self.n_steps - 1 - i
            mask = self._states[t + 1].mask.float()
            current_returns = (
                self._rewards[t] + self.gamma * current_returns * mask
            )
            current_lengths = (
                1 + current_lengths * mask
            )
            sample_returns[t] = current_returns
            sample_lengths[t] = current_lengths

        for e in range(self.n_envs):
            next_state = self._states[self.n_steps][e]
            for i in range(self.n_steps):
                t = self.n_steps - 1 - i
                idx = t * self.n_envs + e
                state = self._states[t][e]
                action = self._actions[t][e]

                sample_states[idx] = state
                sample_actions[idx] = action
                sample_next_states[idx] = next_state

                if not state.mask:
                    next_state = state

        self._states = self._states[self.n_steps:]
        self._actions = self._actions[self.n_steps:]
        self._rewards = self._rewards[self.n_steps:]

        sample_states = State.from_list(sample_states)
        sample_next_states = State.from_list(sample_next_states)

        advantages = (
            sample_returns.view(-1)
            + (self.gamma ** sample_lengths.view(-1))
            * self.v.eval(self.features.eval(sample_next_states))
            - self.v.eval(self.features.eval(sample_states))
        )

        return (
            sample_states,
            sample_actions,
            advantages
        )
