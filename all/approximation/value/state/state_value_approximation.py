from abc import abstractmethod
from all.approximation import Approximation

# pylint: disable=arguments-differ
class StateValueApproximation(Approximation):
    @abstractmethod
    def call(self, state):
        pass

    @abstractmethod
    def update(self, error, state):
        pass

    @abstractmethod
    def gradient(self, state):
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
