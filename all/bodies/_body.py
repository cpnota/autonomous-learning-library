from all.agents import Agent


class Body(Agent):
    """
    A Body wraps a reinforcement learning Agent, altering its inputs and outputs.

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

    def act(self, state):
        return self.process_action(self.agent.act(self.process_state(state)))

    def eval(self, state):
        return self.process_action(self.agent.eval(self.process_state(state)))

    def process_state(self, state):
        return state

    def process_action(self, action):
        return action
