import random
import torch

# https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
class ReplayBuffer:
    def __init__(self, size):
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0

    def store(self, states, action, next_states, reward):
        self._append((states, action, next_states, reward))

    def sample(self, sample_size):
        minibatch = [random.choice(self) for _ in range(0, sample_size)]
        states = [sample[0] for sample in minibatch]
        actions = [sample[1] for sample in minibatch]
        next_states = [sample[2] for sample in minibatch]
        rewards = torch.tensor([sample[3] for sample in minibatch]).float()
        return (states, actions, next_states, rewards)

    def _append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)

    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]

    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        return self.end - self.start

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
