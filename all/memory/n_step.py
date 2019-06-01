import torch
from all.environments import State


class NStepBuffer:
    """A rolling n-step buffer, such that every sample is an n-step rollout."""

    def __init__(self, n, discount_factor=1):
        self.n = n
        self.gamma = discount_factor
        self.i = 0
        self._states = []
        self._actions = []
        self._rewards = []
        self._next_states = []
        self._lengths = []
        self._temp = []

    def __len__(self):
        return len(self._states)

    def store(self, states, actions, rewards):
        # move states that are ready out of temp buffer
        if len(self._temp) == self.n:
            discount = self.gamma ** (self.n - 1)
            for i, v in enumerate(self._temp[0]):
                state, action, reward, last_state, length = v
                if last_state is None:
                    self._store(state, action, reward, last_state, length)
                else:
                    reward += discount * rewards[i]
                    last_state = states[i]
                    self._store(state, action, reward, last_state, length + 1)
            self._temp = self._temp[1:]

        for t, _temp in enumerate(self._temp):
            discount = self.gamma ** (len(self._temp) - 1 - t)
            for i, v in enumerate(_temp):
                state, action, reward, last_state, length = v
                if last_state is not None:
                    reward += discount * rewards[i]
                    last_state = states[i]
                    length += 1
                _temp[i] = (state, action, reward, last_state, length)

        self._temp.append(
            [
                (state, action, reward, next_state, 0)
                for state, action, reward, next_state in zip(
                    states, actions, rewards * 0, states
                )
            ]
        )

        return self

    def sample(self, batch_size):
        if batch_size > len(self):
            raise Exception("Not enough states for batch size!")

        states = self._states[0:batch_size]
        actions = self._actions[0:batch_size]
        actions = torch.tensor(actions, device=actions[0].device)
        next_states = self._next_states[0:batch_size]
        rewards = self._rewards[0:batch_size]
        rewards = torch.tensor(rewards, device=rewards[0].device, dtype=torch.float)
        lengths = self._lengths[0:batch_size]
        lengths = torch.tensor(lengths, device=rewards[0].device, dtype=torch.float)

        self._states = self._states[batch_size:]
        self._actions = self._actions[batch_size:]
        self._next_states = self._next_states[batch_size:]
        self._rewards = self._rewards[batch_size:]
        self._lengths = self._lengths[batch_size:]

        return states, actions, next_states, rewards, lengths

    def _store(self, state, action, reward, next_state, length):
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._next_states.append(next_state)
        self._lengths.append(length)


class NStepBatchBuffer:
    """
    A batch version of the n-step buffer,
    such that the first sample is n-step, second is n-1-step, etc.
    This may be better for on-policy methods.
    """

    def __init__(self, n_steps, n_envs, discount_factor=1):
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
        sample_next_states = [None] * sample_n
        sample_lengths = [0] * sample_n
        sample_returns = [0] * sample_n

        # compute the N-step returns the slow way
        for e in range(self.n_envs):
            for t in range(self.n_steps):
                i = t * self.n_envs + e
                state = self._states[t][e]
                action = self._actions[t][e]
                returns = 0.0
                next_state = state
                sample_length = 0
                if state.mask:
                    for k in range(1, self.n_steps + 1):
                        sample_length += 1
                        next_state = self._states[t + k][e]
                        returns += (self.gamma ** (k - 1)) * self._rewards[t + k][e]
                        if not next_state.mask or t + k == self.n_steps:
                            break
                sample_states[i] = state
                sample_actions[i] = action
                sample_next_states[i] = next_state
                sample_returns[i] = returns
                sample_lengths[i] = sample_length

        self._states = [self._states[-1]]
        self._actions = [self._actions[-1]]
        self._rewards = [self._rewards[-1]]
        sample_returns = torch.tensor(
            sample_returns, device=self._rewards[0].device, dtype=torch.float
        )
        sample_lengths = torch.tensor(
            sample_lengths, device=self._rewards[0].device, dtype=torch.float
        )

        return (
            State.from_list(sample_states),
            sample_actions,
            State.from_list(sample_next_states),
            sample_returns,
            sample_lengths,
        )
