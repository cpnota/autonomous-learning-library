import gymnasium
import torch

from .gym import GymEnvironment
from .gym_wrappers import NoInfoWrapper


class MujocoEnvironment(GymEnvironment):
    """A Mujoco Environment"""

    def __init__(
        self, id, device=torch.device("cpu"), name=None, no_info=True, **gym_make_kwargs
    ):
        super().__init__(id, device=device, name=name, **gym_make_kwargs)
        if no_info:
            self._env = NoInfoWrapper(self._env)
