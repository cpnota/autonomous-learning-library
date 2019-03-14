from abc import abstractmethod
from all.agents import Agent

class Body(Agent):
    """
    A Body wraps a reinforcment learning Agent, altering its inputs and ouputs.

    The Body API is identical to the Agent API from the perspective of the
    rest of the system. This base class is provided only for semantic clarity.
    """
    def __init__(self, agent):
        self._agent = agent
    
    @property
    def agent(self):
        return self._agent

    @agent.setter
    def agent(self, agent):
        self._agent = agent

    def initial(self, state, info=None):
        return self.agent.initial(state, info)

    def act(self, state, reward, info=None):
        return self.agent.act(state, reward, info)

    def terminal(self, reward, info=None):
        return self.agent.terminal(reward, info)
