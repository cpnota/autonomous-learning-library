from abc import ABC, abstractmethod
import torch
from torch.nn import Module


class Preset(ABC, Module):
    def __init__(self, n_envs=None):
        super().__init__()
        self.n_envs = n_envs

    @abstractmethod
    def agent(self, writer=None):
        pass

    @abstractmethod
    def test_agent(self, writer=None):
        pass

    def save(self, filename):
        return torch.save(self, filename)

    def is_parallel(self):
        return self.n_envs is not None
