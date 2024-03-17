import torch

from .gym import GymEnvironment
from .gym_wrappers import NoInfoWrapper


class MujocoEnvironment(GymEnvironment):
    """A Mujoco Environment"""

    def __init__(
        self, id, device=torch.device("cpu"), name=None, no_info=True, **gym_make_kwargs
    ):
        wrap_env = NoInfoWrapper if no_info else None
        super().__init__(
            id, device=device, name=name, wrap_env=wrap_env, **gym_make_kwargs
        )
