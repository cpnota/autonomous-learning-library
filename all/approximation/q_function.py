from abc import ABC, abstractmethod

class QFunction(ABC):
    @abstractmethod
    def __call__(self, state, action=None):
        pass

    @abstractmethod
    def eval(self, state, action=None):
        pass

    @abstractmethod
    def reinforce(self, errors):
        pass
