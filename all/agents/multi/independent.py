from ._multiagent import Multiagent

class IndependentMultiagent(Multiagent):
    def __init__(self, agents):
        self.agents = agents

    def act(self, agent, state):
        return self.agents[agent].act(state)

    def eval(self, agent, state):
        return self.agents[agent].eval(state)
