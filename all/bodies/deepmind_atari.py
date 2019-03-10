import torch
import numpy as np
from .abstract import Body

class DeepmindAtariBody(Body):
    def __init__(
            self,
            agent,
            env,
            frameskip=4
    ):
        self.agent = agent
        self.env = env
        self.frameskip = frameskip
        self._state = None
        self._action = None
        self._skipped_frames = 0
        self._reward = 0

    def initial(self, state, info=None):
        stacked = stack([preprocess(state)] * self.frameskip)
        self._action = self.agent.initial(stacked, info)
        self._state = []
        self._skipped_frames = 0
        return self._action

    def act(self, state, reward, info=None):
        self._state.append(preprocess(state))
        self._reward += clip(reward)
        self._skipped_frames += 1
        if self._skipped_frames == self.frameskip:
            self._action = self.agent.act(
                stack(self._state),
                self._reward,
                info
            )
            self._state = []
            self._reward = 0
            self._skipped_frames = 0
        return self._action

    def terminal(self, reward, info=None):
        self._reward += clip(reward)
        return self.agent.terminal(self._reward, info)

def stack(frames):
    return torch.cat(frames).unsqueeze(0)

def to_grayscale(frame):
    return torch.mean(frame.float(), dim=3).byte()

def downsample(frame):
    return frame[:, ::2, ::2]

def preprocess(frame):
    return to_grayscale(downsample(frame))

def clip(reward):
    return np.sign(reward)
