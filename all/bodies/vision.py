import torch
from all.core import State, StateArray
from ._body import Body

class FrameStack(Body):
    def __init__(self, agent, size=4, lazy=False):
        super().__init__(agent)
        self._frames = []
        self._size = size
        self._lazy = lazy

    def process_state(self, state):
        if not self._frames:
            self._frames = [state.observation] * self._size
        else:
            self._frames = self._frames[1:] + [state.observation]
        if self._lazy:
            return LazyState.from_state(state, self._frames)
        if isinstance(state, StateArray):
            return state.update('observation', torch.cat(self._frames, dim=1))
        return state.update('observation', torch.cat(self._frames, dim=0))

class LazyState(State):
    @classmethod
    def from_state(cls, state, frames):
        state = LazyState(state, device=state.device)
        state['observation'] = frames
        return state

    def __getitem__(self, key):
        if key == 'observation':
            v = dict.__getitem__(self, key)
            if torch.is_tensor(v):
                return v
            return torch.cat(dict.__getitem__(self, key), dim=0)
        return super().__getitem__(key)
