import torch
import numpy as np
from ._body import Body

class ClipRewards(Body):
    def act(self, state, reward):
        if torch.is_tensor(reward):
            return self.agent.act(state, torch.sign(reward))
        return self.agent.act(state, np.sign(reward))
