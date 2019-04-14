from abc import ABC, abstractmethod

class Features(ABC):
    @abstractmethod
    def __call__(self, states):
        pass

    @abstractmethod
    def eval(self, states):
        pass

    @abstractmethod
    def reinforce(self):
        pass
