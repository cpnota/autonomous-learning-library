import torch
from all.environments import State
from ._body import Body

class FrameStack(Body):
    def __init__(self, agent, size=4):
        super().__init__(agent)
        self._frames = []
        self._size = size

    def act(self, state, reward):
        if not self._frames:
            self._frames = [state.raw] * self._size
        else:
            self._frames = self._frames[1:] + [state.raw]
        return self.agent.act(
            State(
                torch.cat(self._frames, dim=1),
                state.mask,
                state.info
            ),
            reward
        )
