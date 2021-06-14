from abc import ABC, abstractmethod
import numpy as np
import torch
from all.core import State
from all.optim import Schedulable
from .segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(ABC):
    @abstractmethod
    def store(self, state, action, reward, next_state):
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
    def __init__(self, size, device='cpu', store_device=None):
        self.buffer = []
        self.capacity = int(size)
        self.pos = 0
        self.device = torch.device(device)
        if store_device is None:
            store_device = self.device
        self.store_device = torch.device(store_device)

    def store(self, state, action, next_state):
        if state is not None and not state.done:
            state = state.to(self.store_device)
            next_state = next_state.to(self.store_device)
            self._add((state, action, next_state))

    def sample(self, batch_size):
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        minibatch = [self.buffer[key] for key in keys]
        return self._reshape(minibatch, torch.ones(batch_size, device=self.device))

    def update_priorities(self, td_errors):
        pass

    def _add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
        self.pos = (self.pos + 1) % self.capacity

    def _reshape(self, minibatch, weights):
        states = State.array([sample[0] for sample in minibatch]).to(self.device)
        if torch.is_tensor(minibatch[0][1]):
            actions = torch.stack([sample[1] for sample in minibatch]).to(self.device)
        else:
            actions = torch.tensor([sample[1] for sample in minibatch], device=self.device)
        next_states = State.array([sample[2] for sample in minibatch]).to(self.device)
        return (states, actions, next_states.reward, next_states, weights)

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)


class PrioritizedReplayBuffer(ExperienceReplayBuffer, Schedulable):
    def __init__(
            self,
            buffer_size,
            alpha=0.6,
            beta=0.4,
            epsilon=1e-5,
            device=torch.device('cpu'),
            store_device=None
    ):
        super().__init__(buffer_size, device=device, store_device=store_device)

        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0
        self._beta = beta
        self._epsilon = epsilon
        self._cache = None

    def store(self, state, action, next_state):
        if state is None or state.done:
            return
        idx = self.pos
        super().store(state, action, next_state)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def sample(self, batch_size):
        beta = self._beta
        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self)) ** (-beta)
            weights.append(weight / max_weight)

        weights = np.array(weights, dtype=np.float32)
        try:
            samples = [self.buffer[idx] for idx in idxes]
        except IndexError as e:
            print('index out of range: ', idxes)
            raise e
        self._cache = idxes
        return self._reshape(samples, torch.from_numpy(weights).to(self.device))

    def update_priorities(self, priorities):
        idxes = self._cache
        _priorities = priorities.detach().cpu().numpy()
        _priorities = np.maximum(_priorities, self._epsilon)
        assert len(idxes) == len(_priorities)
        for idx, priority in zip(idxes, _priorities):
            assert priority > 0
            assert priority < np.inf
            assert 0 <= idx < len(self)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha
            self._max_priority = max(self._max_priority, priority)

    def _sample_proportional(self, batch_size):
        res = []
        p_total = self._it_sum.sum(0, len(self) - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = np.random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res


class NStepReplayBuffer(ReplayBuffer):
    '''Converts any ReplayBuffer into an NStepReplayBuffer'''

    def __init__(
            self,
            steps,
            discount_factor,
            buffer,
    ):
        assert steps >= 1
        assert discount_factor >= 0
        self.steps = steps
        self.discount_factor = discount_factor
        self.buffer = buffer
        self._states = []
        self._actions = []
        self._rewards = []
        self._reward = 0.

    def store(self, state, action, next_state):
        if state is None or state.done:
            return

        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(next_state.reward)
        self._reward += (self.discount_factor ** (len(self._states) - 1)) * next_state.reward

        if len(self._states) == self.steps:
            self._store_next(next_state)

        if next_state.done:
            while self._states:
                self._store_next(next_state)
            self._reward = 0.

    def _store_next(self, next_state):
        self.buffer.store(self._states[0], self._actions[0], next_state.update('reward', self._reward))
        self._reward = self._reward - self._rewards[0]
        self._reward *= self.discount_factor ** -1
        del self._states[0]
        del self._actions[0]
        del self._rewards[0]

    def sample(self, *args, **kwargs):
        return self.buffer.sample(*args, **kwargs)

    def update_priorities(self, *args, **kwargs):
        return self.buffer.update_priorities(*args, **kwargs)

    def __len__(self):
        return len(self.buffer)
