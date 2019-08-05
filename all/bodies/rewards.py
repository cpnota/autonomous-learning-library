import torch
import numpy as np
from ._body import Body

class RewardClipping(Body):
    def act(self, state, reward):
        if isinstance(reward, torch.Tensor):
            self.agent.act(state, torch.sign(reward))
        return self.agent.act(state, np.sign(reward))
