import torch
import numpy as np
from ._body import Body


class ClipRewards(Body):
    def process_state(self, state):
        return state.update('reward', self._clip(state.reward))

    def _clip(self, reward):
        if torch.is_tensor(reward):
            return torch.sign(reward)
        return float(np.sign(reward))
