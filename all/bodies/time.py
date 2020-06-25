import torch
from ._body import Body

class TimeFeature(Body):
    def __init__(self, agent, scale=0.001):
        self.timestep = None
        self.scale = scale
        super().__init__(agent)

    def process_state(self, state):
        if self.timestep is None:
            self.timestep = torch.zeros(len(state), device=state.device)
        features = torch.cat((state.observation, self.scale * self.timestep.view((-1, 1))), dim=1)
        state = state.update('observation', features)
        if torch.is_tensor(state.mask):
            self.timestep = state.mask.float() * (self.timestep + 1)
        else:
            self.timestep = state.mask * (self.timestep + 1)
        return state
