import torch

class NStepBuffer():
    def __init__(self, n, discount_factor=1):
        self.n = n
        self.discount = discount_factor
        self.i = 0
        self.states = []
        self.rewards = []

    def store(self, states, rewards):
        if (self.i == 0):
            self.states = [states]
            self.rewards = [rewards]
            self.i = 1
            # do one thing
        elif (self.i <= self.n):
            self.states.append(states)
            self.rewards.append(rewards)
            self.i += 1
        else:
            raise Exception("Buffer length exceeded: " + self.n)

    def sample(self, _):
        if self.i <= self.n:
            raise Exception("Not enough states received!")
        sample_n = len(self.states[0]) * self.n
        sample_states = [None] * sample_n
        sample_next_states = [None] * sample_n
        sample_returns = torch.zeros(sample_n, device=self.rewards[0].device)
        for i in range(len(self.states[0])):
            returns = 0
            last_state = self.states[self.n][i]
            for j in range(self.n):
                t = self.n - 1 - j
                state = self.states[t][i]
                if state is None:
                    returns = 0
                    last_state = state
                else:
                    returns = self.discount * returns + self.rewards[self.n - j][i]
                index = t * self.n + i
                sample_states[index] = state
                sample_next_states[index] = last_state
                sample_returns[index] = returns
        self.states = [self.states[-1]]
        self.rewards = [self.rewards[-1]]
        self.i = 1
        return (sample_states, sample_next_states, sample_returns)

    def is_full(self):
        return self.i == self.n + 1
