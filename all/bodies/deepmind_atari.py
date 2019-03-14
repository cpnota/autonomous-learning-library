import torch
import numpy as np
from .abstract import Body

NOOP_ACTION = torch.tensor([0])

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

class NoopBody(Body):
    def __init__(self, agent, noop_max):
        self.agent = agent
        self.noop_max = noop_max
        self.noops = 0
        self.actions_taken = 0

    def initial(self, state, info=None):
        self.noops = np.random.randint(self.noop_max)
        self.actions_taken = 0
        return NOOP_ACTION

    def act(self, state, reward, info=None):
        self.actions_taken += 1
        if self.actions_taken < self.noops:
            return NOOP_ACTION
        if self.actions_taken == self.noops:
            return self.agent.initial(state, info)
        return self.agent.act(state, reward)

    def terminal(self, reward, info=None):
        if self.actions_taken >= self.noops:
            return self.agent.terminal(reward, info)
        return # the poor agent never stood a chance

class DeepmindAtariBodyInner(Body):
    def __init__(
            self,
            agent,
            env,
            frameskip=4,
    ):
        self.agent = agent
        self.env = env
        self.frameskip = frameskip
        self._state = None
        self._action = None
        self._reward = 0
        self._info = None
        self._skipped_frames = 0
        self._previous_frame = None
        self._lives = 0

    def initial(self, state, info=None):
        _state = [self._preprocess_initial(state)] * self.frameskip
        self._state = []
        self._action = self.agent.initial(stack(_state), info)
        self._reward = 0
        self._info = info
        self._skipped_frames = 0
        self._lives = self._get_lives()
    
        # fire to start if necessary
        if self._should_fire():
            self._action = torch.tensor([2])
            return torch.tensor([1])

        return self._action

    def act(self, state, reward, info=None):
        if self._lost_life():
            self.terminal(self._reward, self._info)
            return self.initial(state, info)

        self._update_state(state, reward, info)
        if self._should_choose_action():
            self._choose_action()
        return self._action

    def terminal(self, reward, info=None):
        self._reward += clip(reward)
        self._info = info
        return self.agent.terminal(self._reward, self._info)

    def _should_fire(self):
        # pylint: disable=protected-access
        return self.env._env.unwrapped.get_action_meanings()[1] == 'FIRE'

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
        return self._action

    def _preprocess_initial(self, frame):
        self._previous_frame = frame
        return self._preprocess(frame)

    def _preprocess(self, frame):
        deflickered_frame = deflicker(frame, self._previous_frame)
        self._previous_frame = frame
        return to_grayscale(downsample(deflickered_frame))

class DeepmindAtariBody(Body):
    '''
    Enable the Agent to play Atari games DeepMind Style

    Implements the following features:
    1. Frame preprocessing (deflicker + downsample + grayscale)
    2. Frame stacking
    3. Reward clipping (-1, 0, 1)
    4. Episodic lives
    5. Fire on reset
    6. No-op on reset
    '''
    def __init__(self, agent, env, noop_max=30):
        agent = DeepmindAtariBodyInner(agent, env)
        if noop_max > 0:
            agent = NoopBody(agent, noop_max)
        self.agent = agent
