import torch
import numpy as np
from .state import State
from .gym import GymEnvironment
from .atari_wrappers import make_atari, wrap_deepmind

class AtariEnvironment(GymEnvironment):
    def __init__(self, name, **kwargs):
        self._kwargs = kwargs
        env = wrap_deepmind(make_atari(name + 'NoFrameskip-v4'), frame_stack=True, clip_rewards=False)
        super().__init__(env, **kwargs)
        self._name = name

    @property
    def name(self):
        return self._name

    def duplicate(self, n):
        return [AtariEnvironment(self._name, **self._kwargs) for _ in range(n)]

    def _make_state(self, raw, done, info):
        '''Convert numpy array into State'''
        return State(
            torch.from_numpy(
                np.moveaxis(
                    np.array(
                        raw,
                        dtype=self.state_space.dtype
                    ),
                    -1,
                    0
                )
            ).unsqueeze(0).to(self._device),
            self._done_mask if done else self._not_done_mask,
            [info]
        )
