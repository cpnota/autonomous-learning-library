import gym
import torch
from all.core import State
from ._environment import Environment
from .duplicate_env import DuplicateEnvironment
import cloudpickle
gym.logger.set_level(40)


class GymEnvironment(Environment):
    '''
    A wrapper for OpenAI Gym environments (see: https://gym.openai.com).

    This wrapper converts the output of the gym environment to PyTorch tensors,
    and wraps them in a State object that can be passed to an Agent.
    This constructor supports either a string, which will be passed to the
    gym.make(name) function, or a preconstructed gym environment. Note that
    in the latter case, the name property is set to be the whatever the name
    of the outermost wrapper on the environment is.

    Args:
        env: Either a string or an OpenAI gym environment
        name (str, optional): the name of the environment
        device (str, optional): the device on which tensors will be stored
    '''

    def __init__(self, env, device=torch.device('cpu'), name=None):
        if isinstance(env, str):
            self._name = env
            env = gym.make(env)
        else:
            self._name = env.__class__.__name__
        if name:
            self._name = name
        self._env = env
        self._state = None
        self._action = None
        self._reward = None
        self._done = True
        self._info = None
        self._device = device

    @property
    def name(self):
        return self._name

    def reset(self):
        state = self._env.reset(), 0., False, None
        self._state = State.from_gym(state, dtype=self._env.observation_space.dtype, device=self._device)
        return self._state

    def step(self, action):
        self._state = State.from_gym(
            self._env.step(self._convert(action)),
            dtype=self._env.observation_space.dtype,
            device=self._device
        )
        return self._state

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def close(self):
        return self._env.close()

    def seed(self, seed):
        self._env.seed(seed)

    def duplicate(self, n):
        return DuplicateEnvironment([GymEnvironment(cloudpickle.loads(cloudpickle.dumps(self._env)), device=self.device) for _ in range(n)])

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
            if isinstance(self.action_space, gym.spaces.Discrete):
                return action.item()
            if isinstance(self.action_space, gym.spaces.Box):
                return action.cpu().detach().numpy().reshape(-1)
            raise TypeError("Unknown action space type")
        return action
