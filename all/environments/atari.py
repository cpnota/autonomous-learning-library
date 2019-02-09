import gym
import numpy as np
import torch
from .abstract import Environment

def to_grayscale(frame):
    return np.mean(frame, axis=2).astype(np.uint8)

def downsample(frame):
    return frame[::2, ::2]

def preprocess(frame):
    return to_grayscale(downsample(frame))

# pylint: disable=too-many-instance-attributes
class AtariEnvironment(Environment):
    def __init__(self, env):
        if isinstance(env, str):
            self._env = gym.make(env + 'Deterministic-v4')
        else:
            self._env = env

        self._state = None
        self._action = None
        self._reward = None
        self._done = None
        self._info = None

    def reset(self):
        state = self._env.reset()
        self._state = self.process([state] * 4)
        self._done = False
        self._reward = 0
        return self._state

    def step(self, action):
        state = []
        reward = 0
        done = False

        for _ in range(4):
            _state, _reward, _done, info = self._env.step(action.item())
            state.append(_state)
            reward += _reward
            done = _done
            if done:
                break

        self._state = self.process(state) if not done else None
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

    def process(self, state):
        preprocessed = [preprocess(frame) for frame in state]
        return torch.from_numpy(
            np.array(
                preprocessed,
                dtype=self.state_space.dtype
            )
        ).unsqueeze(0)

    @property
    def state_space(self):
        return gym.spaces.Box(low=0, high=255, shape=(4, 105, 80), dtype=np.uint8)

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def state(self):
        return self._state

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
