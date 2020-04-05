import torch
from all.environments import State
from ._body import Body

class FrameStack(Body):
    def __init__(self, agent, size=4, lazy=False):
        super().__init__(agent)
        self._frames = []
        self._size = size
        self._lazy = lazy

    def act(self, state, reward):
        return self.agent.act(self._stack(state), reward)

    def eval(self, state, reward):
        return self.agent.eval(self._stack(state), reward)

    def _stack(self, state):
        if not self._frames:
            self._frames = [state.raw] * self._size
        else:
            self._frames = self._frames[1:] + [state.raw]

        if self._lazy:
            return LazyState(self._frames, state.mask, state.info)

        return State(torch.cat(self._frames, dim=1), state.mask, state.info)

class LazyState(State):
    @property
    def features(self):
        return torch.cat(self._raw, dim=1)

    def __len__(self):
        return len(self._raw[0])
