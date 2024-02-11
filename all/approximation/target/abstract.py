from abc import ABC, abstractmethod


class TargetNetwork(ABC):
    @abstractmethod
    def __call__(self, *inputs):
        pass

    @abstractmethod
    def init(self, model):
        pass

    @abstractmethod
    def update(self):
        pass
