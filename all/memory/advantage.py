import torch
from all.core import State


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
        return len(self._states) * self.n_envs

    def store(self, states, actions, rewards):
        if states is None:
            return
        if not self._states:
            self._states = [states]
            self._actions = [actions]
            self._rewards = [rewards]
        elif len(self._states) < self.n_steps:
            self._states.append(states)
            self._actions.append(actions)
            self._rewards.append(rewards)
        else:
            raise Exception("Buffer length exceeded: " + str(self.n_steps))

    def advantages(self, states):
        if len(self) < self.n_steps * self.n_envs:
            raise Exception("Not enough states received!")

        self._states.append(states)
        rewards, lengths = self._compute_returns()
        states, actions, next_states = self._summarize_transitions()
        advantages = self._compute_advantages(states, rewards, next_states, lengths)
        self._clear_buffers()

        return (
            states,
            actions,
            advantages
        )

    def _compute_returns(self):
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

        return sample_returns, sample_lengths

    def _summarize_transitions(self):
        sample_n = self.n_envs * self.n_steps
        sample_states = [None] * sample_n
        sample_actions = [None] * sample_n
        sample_next_states = [None] * sample_n

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

        return (
            State.array(sample_states),
            torch.stack(sample_actions),
            State.array(sample_next_states)
        )

    def _compute_advantages(self, states, rewards, next_states, lengths):
        return (
            rewards.view(-1)
            + (self.gamma ** lengths.view(-1))
            * self.v.target(self.features.target(next_states)).view(-1)
            - self.v.eval(self.features.eval(states)).view(-1)
        )

    def _clear_buffers(self):
        self._states = []
        self._actions = []
        self._rewards = []
