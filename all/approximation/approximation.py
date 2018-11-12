from abc import ABC, abstractmethod

class Approximation(ABC):
    @abstractmethod
    def call(self, *args):
        pass

    @abstractmethod
    def update(self, error, *args):
        pass

    @abstractmethod
    def gradient(self, *args):
        pass

    @abstractmethod
    def apply(self, gradient):
        pass

    @property
    @abstractmethod
    def parameters(self):
        pass

    @parameters.setter
    @abstractmethod
    def parameters(self, parameters):
        pass
