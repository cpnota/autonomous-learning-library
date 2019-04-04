from abc import ABC, abstractmethod
import random
import numpy as np
import torch
from .segment_tree import SumSegmentTree, MinSegmentTree

class ReplayBuffer(ABC):
    @abstractmethod
    def store(self, state, action, next_state, reward):
        '''Store the transition in the buffer'''

    @abstractmethod
    def sample(self, batch_size):
        '''Sample from the stored transitions'''

    @abstractmethod
    def update_priorities(self, indexes, td_errors):
        '''Update priorities based on the TD error'''


# Adapted from:
# https://github.com/Shmuma/ptan/blob/master/ptan/experience.py
class ExperienceReplayBuffer(ReplayBuffer):
    def __init__(self, size, device=torch.device('cpu')):
        self.buffer = []
        self.capacity = size
        self.pos = 0
        self.device = device

    def store(self, states, action, next_states, reward):
        self._add((states, action, next_states, reward))

    def sample(self, batch_size):
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        minibatch = [self.buffer[key] for key in keys]
        return self._reshape(minibatch, torch.ones(batch_size, device=self.device, dtype=torch.half))

    def update_priorities(self, td_errors):
        pass

    def _add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
        self.pos = (self.pos + 1) % self.capacity

    def _reshape(self, minibatch, weights):
        states = [sample[0] for sample in minibatch]
        actions = [sample[1] for sample in minibatch]
        next_states = [sample[2] for sample in minibatch]
        rewards = torch.tensor([sample[3] for sample in minibatch], device=self.device, dtype=torch.half)
        return (states, actions, next_states, rewards, weights)

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

class PrioritizedReplayBuffer(ExperienceReplayBuffer):
    def __init__(
            self,
            buffer_size,
            alpha=0.6,
            beta=0.4,
            final_beta_frame=100000,
            epsilon=1e-5,
            device=torch.device('cpu')
    ):
        super().__init__(buffer_size, device=device)

        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        self._beta = beta
        self._final_beta_frame = final_beta_frame
        self._epsilon = epsilon
        self._frames = 0
        self._cache = None

    def store(self, states, action, next_states, reward):
        self._add((states, action, next_states, reward))

    def sample(self, batch_size):
        beta = min(1.0, self._beta + self._frames
                   * (1.0 - self._beta) / self._final_beta_frame)
        self._frames += 1
        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self)) ** (-beta)
            weights.append(weight / max_weight)

        weights = np.array(weights, dtype=np.float32)
        samples = [self.buffer[idx] for idx in idxes]
        self._cache = idxes
        return self._reshape(samples, torch.from_numpy(weights).half().to(self.device))

    def update_priorities(self, td_errors):
        idxes = self._cache
        _td_errors = td_errors.detach().numpy()
        priorities = list(np.abs(_td_errors) + self._epsilon)
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

    def _add(self, *args, **kwargs):
        idx = self.pos
        super()._add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res
