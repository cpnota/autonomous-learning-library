import gym
import numpy as np
import torch
from .abstract import Environment

# Easy version of pong.
class PongEnvironment(Environment):
    def __init__(self):
        self._env = gym.make('PongDeterministic-v4')
        self._state = None
        self._action = None
        self._reward = None
        self._done = None
        self._info = None

    def reset(self):
        state = self._env.reset()
        self._state = torch.cat([self.process(state)] * 4).unsqueeze(0)
        self._done = False
        self._reward = 0
        return self._state

    def step(self, action):
        frame, reward, done, info = self._env.step(self.map_action(action))

        # end on every score, to make game much easier
        if reward != 0:
            done = True

        self._state = self.update_state(frame) if not done else None
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

    def update_state(self, frame):
        return torch.cat((self._state[0, 1:], self.process(frame))).unsqueeze(0)

    def process(self, frame):
        return torch.from_numpy(
            np.array(
                preprocess(frame),
                dtype=self.state_space.dtype
            )
        ).unsqueeze(0)

    def map_action(self, action):
        i = action.item()
        if i == 0:
            # none
            return 0
        if i == 1:
            # right
            return 3
        if i == 2:
            # left
            return 2
        raise Exception('Unknown action:', i)

    @property
    def state_space(self):
        return gym.spaces.Box(low=0, high=255, shape=(4, 105, 80), dtype=np.uint8)

    @property
    def action_space(self):
        return gym.spaces.Discrete(3)

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

def to_grayscale(frame):
    return np.mean(frame, axis=2).astype(np.uint8)

def downsample(frame):
    return frame[::2, ::2]

def preprocess(frame):
    return to_grayscale(downsample(frame))
