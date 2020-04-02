import torch
from all.environments import State
from ._body import Body

class TimeFeature(Body):
    def __init__(self, agent, scale=0.001):
        self.timestep = None
        self.scale = scale
        super().__init__(agent)

    def act(self, state, reward):
        return self.agent.act(self._append_time_feature(state), reward)

    def eval(self, state, reward):
        return self.agent.eval(self._append_time_feature(state), reward)

    def _append_time_feature(self, state):
        if self.timestep is None:
            self.timestep = torch.zeros(len(state), device=state.features.device)
        features = torch.cat((state.features, self.scale * self.timestep.view((-1, 1))), dim=1)
        state = State(features, state.mask, state.info)
        self.timestep = state.mask.float() * (self.timestep + 1)
        return state
