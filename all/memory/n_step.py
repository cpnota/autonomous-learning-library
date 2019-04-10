import torch


class NStepBuffer():
    def __init__(self, n, discount_factor=1):
        self.n = n
        self.gamma = discount_factor
        self.i = 0
        self._states = []
        self._rewards = []
        self._next_states = []
        self._lengths = []
        self._temp = []

    def __len__(self):
        return len(self._states)

    def store(self, states, rewards):
        # move states that are ready out of temp buffer
        if len(self._temp) == self.n:
            discount = self.gamma ** (self.n - 1)
            for i, v in enumerate(self._temp[0]):
                state, reward, last_state, length = v
                if last_state is None:
                    self._store(state, reward, last_state, length)
                else:
                    reward += discount * rewards[i]
                    last_state = states[i]
                    self._store(state, reward, last_state, length + 1)
            self._temp = self._temp[1:]

        for t, _temp in enumerate(self._temp):
            discount = self.gamma ** (len(self._temp) - 1 - t)
            for i, v in enumerate(_temp):
                state, reward, last_state, length = v
                if last_state is not None:
                    reward += discount * rewards[i]
                    last_state = states[i]
                    length += 1
                _temp[i] = (state, reward, last_state, length)

        self._temp.append([
            (state, reward, next_state, 0)
            for state, reward, next_state
            in zip(states, rewards * 0, states)
        ])

        return self

    def sample(self, batch_size):
        if batch_size > len(self):
            raise Exception("Not enough states for batch size!")

        states = self._states[0:batch_size]
        next_states = self._next_states[0:batch_size]
        rewards = self._rewards[0:batch_size]
        rewards = torch.tensor(rewards, device=rewards[0].device, dtype=torch.float)
        lengths = self._lengths[0:batch_size]
        lengths = torch.tensor(lengths, device=rewards[0].device)

        self._states = self._states[batch_size:]
        self._next_states = self._next_states[batch_size:]
        self._rewards = self._rewards[batch_size:]
        self._lengths = self._lengths[batch_size:]

        return states, next_states, rewards, lengths


    def _store(self, state, reward, next_state, length):
        self._states.append(state)
        self._rewards.append(reward)
        self._next_states.append(next_state)
        self._lengths.append(length)
