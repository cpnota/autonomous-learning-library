from abc import ABC, abstractmethod

class ValueFunction(ABC):
    @abstractmethod
    def __call__(self, states):
        pass

    @abstractmethod
    def eval(self, states):
        pass

    @abstractmethod
    def train(self, targets):
        pass
