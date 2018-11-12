from abc import ABC, abstractmethod

class Basis(ABC):
    @abstractmethod
    def features(self, args):
        pass

    @property
    @abstractmethod
    def num_features(self):
        pass
