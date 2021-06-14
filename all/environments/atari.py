import gym
from .gym import GymEnvironment
from .atari_wrappers import (
    NoopResetEnv,
    MaxAndSkipEnv,
    FireResetEnv,
    WarpFrame,
    LifeLostEnv,
)
from all.core import State
from .duplicate_env import DuplicateEnvironment


class AtariEnvironment(GymEnvironment):
    def __init__(self, name, *args, **kwargs):
        # need these for duplication
        self._args = args
        self._kwargs = kwargs
        # construct the environment
        env = gym.make(name + "NoFrameskip-v4")
        # apply a subset of wrappers
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = WarpFrame(env)
        env = LifeLostEnv(env)
        # initialize
        super().__init__(env, *args, **kwargs)
        self._name = name

    @property
    def name(self):
        return self._name

    def reset(self):
        state = self._env.reset(), 0., False, {'life_lost': False}
        self._state = State.from_gym(state, dtype=self._env.observation_space.dtype, device=self._device)
        return self._state

    def duplicate(self, n):
        return DuplicateEnvironment([
            AtariEnvironment(self._name, *self._args, **self._kwargs) for _ in range(n)
        ])
