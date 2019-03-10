from abc import abstractmethod
from all.agents import Agent

class Body(Agent):
    """
    A Body wraps a reinforcment learning Agent, altering its inputs and ouputs.

    The Body API is identical to the Agent API from the perspective of the
    rest of the system. This base class is provided only for semantic clarity.
    """

    @abstractmethod
    def initial(self, state, info=None):
        """See Agent"""

    @abstractmethod
    def act(self, state, reward, info=None):
        """See Agent"""

    @abstractmethod
    def terminal(self, reward, info=None):
        """See Agent"""
