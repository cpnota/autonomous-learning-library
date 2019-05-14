import torch


class NStepBuffer():
    '''A rolling n-step buffer, such that every sample is an n-step rollout.'''
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

        self._temp.append([
            (state, action, reward, next_state, 0)
            for state, action, reward, next_state
            in zip(states, actions, rewards * 0, states)
        ])

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

class NStepBatchBuffer():
    '''
    A batch version of the n-step buffer, such that the first sample is n-step, second is n-1-step, etc.
    This may be better for on-policy methods.
    '''
    def __init__(self, n, batch_size, discount_factor=1):
        self.n = n
        self.batch_size = batch_size
        self.gamma = discount_factor
        self.i = 0
        self._states = []
        self._rewards = []

    def __len__(self):
        return len(self._states)

    def store(self, states, rewards):
        if self.i == 0:
            self._states = [states]
            self._rewards = [rewards]
            self.i = 1
        elif self.i <= self.batch_size:
            self._states.append(states)
            self._rewards.append(rewards)
            self.i += 1
        else:
            raise Exception("Buffer length exceeded: " + self.n)

    def sample(self, _):
        if self.i <= self.batch_size:
            raise Exception("Not enough states received!")

        n_envs = len(self._states[0])
        sample_n = n_envs * self.batch_size
        sample_states = [None] * sample_n
        sample_next_states = [None] * sample_n
        sample_returns = torch.zeros(sample_n, device=self._rewards[0].device)

        # compute the N-step returns the slow way
        for e in range(n_envs):
            for t in range(self.batch_size):
                i = t * n_envs + e
                state = self._states[t][e]
                returns = 0.
                next_state = None
                if state is not None:
                    for k in range(1, self.n + 1):
                        next_state = self._states[t + k][e]
                        returns += (self.gamma ** (k - 1)) * \
                            self._rewards[t + k][e]
                        if next_state is None or t + k == self.batch_size:
                            break
                sample_states[i] = state
                sample_next_states[i] = next_state
                sample_returns[i] = returns

        self._states = [self._states[-1]]
        self._rewards = [self._rewards[-1]]
        self.i = 1
        return (sample_states, sample_next_states, sample_returns)
