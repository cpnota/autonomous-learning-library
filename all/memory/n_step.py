import torch


class NStepBuffer():
    def __init__(self, n, buffer_length, discount_factor=1):
        self.n = n
        self.buffer_length = buffer_length
        self.gamma = discount_factor
        self.i = 0
        self.states = []
        self.rewards = []

    def store(self, states, rewards):
        if self.i == 0:
            self.states = [states]
            self.rewards = [rewards]
            self.i = 1
            # do one thing
        elif self.i <= self.buffer_length:
            self.states.append(states)
            self.rewards.append(rewards)
            self.i += 1
        else:
            raise Exception("Buffer length exceeded: " + self.n)

    def sample(self, _):
        if self.i <= self.buffer_length:
            raise Exception("Not enough states received!")

        n_envs = len(self.states[0])
        sample_n = n_envs * self.buffer_length
        sample_states = [None] * sample_n
        sample_next_states = [None] * sample_n
        sample_returns = torch.zeros(sample_n, device=self.rewards[0].device)
        sample_lengths = torch.zeros(sample_n, device=self.rewards[0].device)

        # compute the N-step returns the slow way
        for e in range(n_envs):
            for t in range(self.buffer_length):
                i = t * n_envs + e
                state = self.states[t][e]
                returns = 0.
                next_state = None
                sample_length = 0
                if state is not None:
                    for k in range(1, self.n + 1):
                        sample_length = k
                        next_state = self.states[t + k][e]
                        returns += (self.gamma ** (k - 1)) * \
                            self.rewards[t + k][e]
                        if next_state is None or t + k == self.buffer_length:
                            break
                sample_states[i] = state
                sample_next_states[i] = next_state
                sample_returns[i] = returns
                sample_lengths[i] = sample_length

        self.states = [self.states[-1]]
        self.rewards = [self.rewards[-1]]
        self.i = 1
        return (sample_states, sample_next_states, sample_returns, sample_lengths)

    def is_full(self):
        return self.i == self.buffer_length + 1
