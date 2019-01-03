from abc import abstractmethod
from all.approximation import Approximation

# pylint: disable=arguments-differ
class StateValue(Approximation):
    @abstractmethod
    def __call__(self, state):
        pass

    @abstractmethod
    def update(self, error, state):
        pass
