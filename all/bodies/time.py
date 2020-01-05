import torch
from all.environments import State
from ._body import Body

class TimeFeature(Body):
    def __init__(self, agent):
        self.timestep = None
        super().__init__(agent)

    def act(self, state, reward):
        if self.timestep is None:
            self.timestep = torch.zeros(len(state))
        features = torch.cat((state.features, self.timestep.view((-1, 1))), dim=1)
        state = State(features, state.mask, state.info)
        self.timestep = state.mask * (self.timestep + 1)
        return self.agent.act(state, reward)
