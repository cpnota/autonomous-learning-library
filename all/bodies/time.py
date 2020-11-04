import torch
from all.core import StateArray
from ._body import Body


class TimeFeature(Body):
    def __init__(self, agent, scale=0.001):
        self.timestep = None
        self.scale = scale
        super().__init__(agent)

    def process_state(self, state):
        if isinstance(state, StateArray):
            if self.timestep is None:
                self.timestep = torch.zeros(state.shape, device=state.device)
            observation = torch.cat((state.observation, self.scale * self.timestep.view(-1, 1)), dim=1)
            state = state.update('observation', observation)
            self.timestep = state.mask.float() * (self.timestep + 1)
            return state

        if self.timestep is None:
            self.timestep = 0
        state.update('timestep', self.timestep)
        observation = torch.cat((
            state.observation,
            torch.tensor(self.scale * self.timestep, device=state.device).view(-1)
        ), dim=0)
        state = state.update('observation', observation)
        self.timestep = state.mask * (self.timestep + 1)
        return state
