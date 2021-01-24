from ._multiagent import Multiagent


class IndependentMultiagent(Multiagent):
    def __init__(self, agents):
        self.agents = agents

    def act(self, state):
        return self.agents[state['agent']].act(state)
