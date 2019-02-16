import gym
import numpy as np
import torch
from .abstract import Environment

# pylint: disable=too-many-instance-attributes
class AtariEnvironment(Environment):
    def __init__(self, env, episodic_lives=True):
        if isinstance(env, str):
            self._env = gym.make(env + 'Deterministic-v4')
        else:
            self._env = env

        self._state = None
        self._action = None
        self._reward = None
        self._done = None
        self._info = None
        self._should_reset = None

        # for episodic life
        self.episodic_lives = episodic_lives
        self._lives = None

    def reset(self):
        state = self._env.reset()
        self._state = torch.cat([self.process(state)] * 4).unsqueeze(0)
        self._done = False
        self._reward = 0
        self._should_reset = False
        self._lives = self._env.unwrapped.ale.lives()
        return self._state

    def step(self, action):
        frame, reward, done, info = self._env.step(action.item())
        self._state = self.update_state(frame) if not done else None
        self._action = action
        self._reward = reward
        self._done = done
        self._info = info
        self._should_reset = done

        if self.episodic_lives:
            lives = self.env.unwrapped.ale.lives()
            if 0 < lives < self._lives:
                self._done = True
            if lives < self._lives:
                self._reward = -1
            self._lives = lives

        return self._state, self._reward, self._done, self._info

    def render(self):
        return self._env.render()

    def close(self):
        return self._env.close()

    def seed(self, seed):
        self._env.seed(seed)

    def update_state(self, frame):
        return torch.cat((self._state[0, 1:], self.process(frame))).unsqueeze(0)

    def process(self, frame):
        return torch.from_numpy(
            np.array(
                preprocess(frame),
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
    def should_reset(self):
        return self._should_reset

    @property
    def env(self):
        return self._env

def to_grayscale(frame):
    return np.mean(frame, axis=2).astype(np.uint8)

def downsample(frame):
    return frame[::2, ::2]

def preprocess(frame):
    return to_grayscale(downsample(frame))
