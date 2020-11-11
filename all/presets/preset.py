from abc import ABC, abstractmethod
from torch.nn import Module


class Preset(ABC, Module):
    @abstractmethod
    def agent(self, writer=None):
        pass

    @abstractmethod
    def test_agent(self, writer=None):
        pass

    def save(self, filename):
        return torch.save(self, filename)
