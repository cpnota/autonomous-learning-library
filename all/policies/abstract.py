from abc import abstractmethod

# pylint: disable=arguments-differ
class Policy():
    @abstractmethod
    def __call__(self, state, action=None, prob=False):
        pass

    @abstractmethod
    def reinforce(self, errors):
        pass
