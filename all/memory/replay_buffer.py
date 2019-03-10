from abc import ABC, abstractmethod
import numpy as np
import torch

class ReplayBuffer(ABC):
    @abstractmethod
    def store(self, state, action, next_state, reward):
        '''Store the transition in the buffer'''

    @abstractmethod
    def sample(self, batch_size):
        '''Sample from the stored transitions'''

# Adapted from:
# https://github.com/Shmuma/ptan/blob/master/ptan/experience.py
class ExperienceReplayBuffer(ReplayBuffer):
    def __init__(self, size):
        self.buffer = []
        self.capacity = size
        self.pos = 0

    def store(self, states, action, next_states, reward):
        self._add((states, action, next_states, reward))

    def sample(self, batch_size):
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        minibatch = [self.buffer[key] for key in keys]
        return self._reshape(minibatch)

    def _add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
            self.pos = (self.pos + 1) % self.capacity

    def _reshape(sefl, minibatch):
        states = [sample[0] for sample in minibatch]
        actions = [sample[1] for sample in minibatch]
        next_states = [sample[2] for sample in minibatch]
        rewards = torch.tensor([sample[3] for sample in minibatch]).float()
        return (states, actions, next_states, rewards)

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

