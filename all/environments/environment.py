from abc import ABC, abstractmethod

class Environment(ABC):
    """
    An Environment.
    """

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def close(self):
        pass

    @property
    @abstractmethod
    def state(self):
        pass

    @property
    @abstractmethod
    def action(self):
        pass

    @property
    @abstractmethod
    def reward(self):
        pass

    @property
    @abstractmethod
    def done(self):
        pass

    @property
    def info(self):
        return None
