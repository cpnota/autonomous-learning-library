from abc import abstractmethod
from all.approximation import Approximation

# pylint: disable=arguments-differ
class ActionValueApproximation(Approximation):
    @abstractmethod
    def call(self, state, action=None):
        pass

    @abstractmethod
    def update(self, error, state, action):
        pass

    @abstractmethod
    def gradient(self, state, action):
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
