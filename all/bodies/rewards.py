import torch
import numpy as np
from ._body import Body

class ClipRewards(Body):
    def act(self, state, reward):
        return self.agent.act(state, self._clip(reward))

    def eval(self, state, reward):
        return self.agent.eval(state, self._clip(reward))

    def _clip(self, reward):
        if torch.is_tensor(reward):
            return torch.sign(reward)
        return np.sign(reward)
