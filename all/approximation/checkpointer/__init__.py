from abc import abstractmethod, ABC
import torch

class Checkpointer(ABC):
    @abstractmethod
    def init(self, model, filename):
        pass

    @abstractmethod
    def __call__(self):
        pass


class DummyCheckpointer(Checkpointer):
    def init(self, *inputs):
        pass

    def __call__(self):
        pass


class PeriodicCheckpointer(Checkpointer):
    def __init__(self, frequency):
        self.frequency = frequency
        self._updates = 1
        self._filename = None
        self._model = None

    def init(self, model, filename):
        self._model = model
        self._filename = filename

    def __call__(self):
        if self._updates % self.frequency == 0:
            torch.save(self._model, self._filename)
        self._updates += 1
