from abc import abstractmethod
from all.approximation import Approximation

# pylint: disable=arguments-differ
class Policy(Approximation):
    @abstractmethod
    def __call__(self, state, action=None, prob=False):
        pass

    @abstractmethod
    def update(self, error, state, action):
        pass
