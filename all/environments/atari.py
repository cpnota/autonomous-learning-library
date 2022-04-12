import gym
import torch
from all.core import State
from .duplicate_env import DuplicateEnvironment
from .atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    FireResetEnv,
    WarpFrame,
    LifeLostEnv,
)
from ._environment import Environment


class AtariEnvironment(Environment):
    def __init__(self, name, device='cpu'):

        # construct the environment
        env = gym.make(name + "NoFrameskip-v4")

        # apply a subset of wrappers
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpFrame(env)
        env = LifeLostEnv(env)

        # initialize member variables
        self._env = env
        self._name = name
        self._state = None
        self._action = None
        self._reward = None
        self._done = True
        self._info = None
        self._device = device

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
        return DuplicateEnvironment([AtariEnvironment(self._name, device=self._device) for _ in range(n)])

    @property
    def name(self):
        return self._name

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
            return action.item()
        return action
