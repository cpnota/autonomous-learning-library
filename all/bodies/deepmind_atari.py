import torch
import numpy as np
from .abstract import Body

class DeepmindAtariBody(Body):
    '''
    Enable the Agent to play Atari games DeepMind Style

    Implements the following features:
    1. Frame preprocessing (downsample + grayscale)
    2. Frame stacking
    3. Reward clipping (-1, 0, 1)
    3. Episodic lives (not implemented)
    4. Fire on reset (not implemented)
    3. Frame preprocessing (downsample + grayscale)
    4.
    '''

    def __init__(
            self,
            agent,
            env,
            frameskip=4,
            noop_max=30
    ):
        self.agent = agent
        self.env = env
        self.frameskip = frameskip
        self.noop_max = noop_max
        self._state = None
        self._action = None
        self._reward = 0
        self._info = None
        self._skipped_frames = 0
        self._previous_frame = None
        self._lives = 0

    def initial(self, state, info=None):
        self._set_initial_state(state, info)
        self._choose_action()
        return self._action

    def act(self, state, reward, info=None):
        self._update_state(state, reward, info)
        if self._lost_life():
            self.terminal(0, info)
            self.initial(state, info)
        if self._should_choose_action():
            self._choose_action()
        return self._action

    def terminal(self, reward, info=None):
        self._reward += clip(reward)
        self._info = info
        return self.agent.terminal(self._reward, self._info)

    def _set_initial_state(self, state, info):
        self._state = [self._preprocess_initial(state)] * self.frameskip
        self._reward = 0
        self._info = info
        self._lives = self._get_lives()

    def _update_state(self, state, reward, info):
        self._state.append(self._preprocess(state))
        self._reward += clip(reward)
        self._info = info
        self._skipped_frames += 1

    def _lost_life(self):
        lives = self._get_lives()
        return lives < self._lives and lives > 0

    def _get_lives(self):
        # pylint: disable=protected-access
        return self.env._env.unwrapped.ale.lives()

    def _should_choose_action(self):
        return self._skipped_frames == self.frameskip

    def _choose_action(self):
        self._action = self.agent.act(
            stack(self._state),
            self._reward,
            self._info
        )
        self._state = []
        self._reward = 0
        self._skipped_frames = 0

    def _preprocess_initial(self, frame):
        self._previous_frame = frame
        return self._preprocess(frame)

    def _preprocess(self, frame):
        deflickered_frame = deflicker(frame, self._previous_frame)
        self._previous_frame = frame
        return to_grayscale(downsample(deflickered_frame))

def stack(frames):
    return torch.cat(frames).unsqueeze(0)

def to_grayscale(frame):
    return torch.mean(frame.float(), dim=3).byte()

def downsample(frame):
    return frame[:, ::2, ::2]

def deflicker(frame1, frame2):
    return torch.max(frame1, frame2)

def clip(reward):
    return np.sign(reward)
