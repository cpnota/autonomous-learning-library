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

    def act(self, state, reward):
        return self.agent.act(state, reward)

    def eval(self, state, reward):
        return self.agent.eval(state, reward)
