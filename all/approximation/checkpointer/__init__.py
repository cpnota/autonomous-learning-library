import warnings
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
        # Some builds of pytorch throw this unhelpful warning.
        # We can safely disable it.
        # https://discuss.pytorch.org/t/got-warning-couldnt-retrieve-source-code-for-container/7689/7
        warnings.filterwarnings("ignore", message="Couldn't retrieve source code")

    def __call__(self):
        if self._updates % self.frequency == 0:
            torch.save(self._model, self._filename)
        self._updates += 1
