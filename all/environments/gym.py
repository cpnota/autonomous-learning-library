import gymnasium
import torch
from all.core import State
from ._environment import Environment
from .duplicate_env import DuplicateEnvironment

gymnasium.logger.set_level(40)


class GymEnvironment(Environment):
    """
    A wrapper for OpenAI Gym environments (see: https://gymnasium.openai.com).

    This wrapper converts the output of the gym environment to PyTorch tensors,
    and wraps them in a State object that can be passed to an Agent.
    This constructor supports either a string, which will be passed to the
    gymnasium.make(name) function, or a preconstructed gym environment. Note that
    in the latter case, the name property is set to be the whatever the name
    of the outermost wrapper on the environment is.

    Args:
        env: Either a string or an OpenAI gym environment
        name (str, optional): the name of the environment
        device (str, optional): the device on which tensors will be stored
        legacy_gym (str, optional): If true, calls gym.make() instead of gymnasium.make()
        **gym_make_kwargs: kwargs passed to gymnasium.make(id, **gym_make_kwargs)
    """

    def __init__(
        self,
        id,
        device=torch.device("cpu"),
        name=None,
        legacy_gym=False,
        **gym_make_kwargs
    ):
        if legacy_gym:
            import gym

            self._gym = gym
        else:
            self._gym = gymnasium
        self._env = self._gym.make(id, **gym_make_kwargs)
        self._id = id
        self._name = name if name else id
        self._state = None
        self._action = None
        self._reward = None
        self._done = True
        self._info = None
        self._device = device

    @property
    def name(self):
        return self._name

    def reset(self, **kwargs):
        self._state = State.from_gym(
            self._env.reset(**kwargs),
            dtype=self._env.observation_space.dtype,
            device=self._device,
        )
        return self._state

    def step(self, action):
        self._state = State.from_gym(
            self._env.step(self._convert(action)),
            dtype=self._env.observation_space.dtype,
            device=self._device,
        )
        return self._state

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def close(self):
        return self._env.close()

    def seed(self, seed):
        self._env.seed(seed)

    def duplicate(self, n):
        return DuplicateEnvironment(
            [
                GymEnvironment(self._id, device=self.device, name=self._name)
                for _ in range(n)
            ]
        )

    @property
    def state_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def state(self):
        return self._state

    @property
    def env(self):
        return self._env

    @property
    def device(self):
        return self._device

    def _convert(self, action):
        if torch.is_tensor(action):
            if isinstance(self.action_space, self._gym.spaces.Discrete):
                return action.item()
            if isinstance(self.action_space, self._gym.spaces.Box):
                return action.cpu().detach().numpy().reshape(-1)
            raise TypeError("Unknown action space type")
        return action
