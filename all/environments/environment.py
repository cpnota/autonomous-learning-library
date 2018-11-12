from abc import ABC, abstractmethod

class Environment(ABC):
    """
    A reinforcement learning Environment.

    In reinforcement learning, an Agent learns by interacting with an Environment.
    An Environment defines the dynamics of a particular problem:
    the states, the actions, the transitions between states, and the rewards given to the agent.
    Environments are often used to benchmark reinforcement learning agents,
    or to define real problems that the user hopes to solve using reinforcement learning.
    """

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def render(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @property
    @abstractmethod
    def action_space(self):
        pass

    @property
    @abstractmethod
    def state_space(self):
        pass

    @property
    def observation_space(self):
        return self.state_space

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
