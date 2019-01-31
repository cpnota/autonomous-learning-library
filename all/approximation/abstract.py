from abc import ABC, abstractmethod

class Approximation(ABC):
    @abstractmethod
    def __call__(self, *args):
        pass

    @abstractmethod
    def update(self, error, *args):
        pass
