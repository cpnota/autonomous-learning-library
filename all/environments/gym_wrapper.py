import gym
import numpy as np
import torch
from .abstract import Environment

def preprocess(preprocessors):
    def _(frame):
        if not preprocessors:
            return frame
        return preprocess(preprocessors[1:])(preprocessors[0](frame))
    return _

# pylint: disable=too-many-instance-attributes
class GymWrapper(Environment):
    def __init__(self, env, preprocessors=None):
        if isinstance(env, str):
            self._env = gym.make(env)
        else:
            self._env = env

        self._state = None
        self._action = None
        self._reward = None
        self._done = None
        self._info = None
        preprocessors = [] if preprocessors is None else preprocessors
        self.preprocess = preprocess(preprocessors)

    def reset(self):
        state = self._env.reset()
        self.state = state
        self._done = False
        self._reward = 0
        return self._state

    def step(self, action):
        state, reward, done, info = self._env.step(action.item())
        self.state = state if not done else None
        self._action = action
        self._reward = reward
        self._done = done
        self._info = info
        return self._state, self._reward, self._done, self._info

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()

    def seed(self, seed):
        self._env.seed(seed)

    @property
    def state_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        if value is None:
            self._state = None
        else:
            # Somewhat tortured method of
            # ensuring that the tensor
            # is of the correct type.
            self._state = torch.from_numpy(
                np.array(
                    self.preprocess(value),
                    dtype=self.state_space.dtype
                )
            ).unsqueeze(0)

    @property
    def action(self):
        return self._action

    @property
    def reward(self):
        return self._reward

    @property
    def done(self):
        return self._done

    @property
    def info(self):
        return self._info

    @property
    def env(self):
        return self._env
