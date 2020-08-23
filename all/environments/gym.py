import gym
import torch
from all.core import State
from .abstract import Environment
gym.logger.set_level(40)

class GymEnvironment(Environment):
    def __init__(self, env, device=torch.device('cpu')):
        
        if isinstance(env, str):
            self._name = env
            env = gym.make(env)
        else:
            self._name = env.__class__.__name__

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
        self._state = State.from_gym(self._env.reset(), device=self._device)
        return self._state

    def step(self, action):
        self._state = State.from_gym(self._env.step(self._convert(action)), device=self._device)
        return self._state

    def render(self, **kwargs):
        return self._env.render(**kwargs)

    def close(self):
        return self._env.close()

    def seed(self, seed):
        self._env.seed(seed)

    def duplicate(self, n):
        return [GymEnvironment(self._name, device=self.device) for _ in range(n)]

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
