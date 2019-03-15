import torch
import numpy as np
from .abstract import Body

NOOP_ACTION = torch.tensor([0])

class NoopBody(Body):
    def __init__(self, agent, noop_max):
        super().__init__(agent)
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
        return  # the poor agent never stood a chance

class AtariVisionPreprocessor(Body):
    def __init__(self, agent, deflicker=True):
        super().__init__(agent)
        self.deflicker = deflicker
        self._previous_frame = None

    def initial(self, frame, info=None):
        if self.deflicker:
            self._previous_frame = frame
        return self.agent.initial(preprocess(frame), info)

    def act(self, frame, reward, info=None):
        if self.deflicker:
            frame, self._previous_frame = torch.max(
                frame, self._previous_frame), frame
        return self.agent.act(preprocess(frame), reward, info)

def preprocess(frame):
    return to_grayscale(downsample(frame))

def stack(frames):
    return torch.cat(frames).unsqueeze(0)

def to_grayscale(frame):
    return torch.mean(frame.float(), dim=3).byte()

def downsample(frame):
    return frame[:, ::2, ::2]

class RewardClipping(Body):
    def act(self, state, reward, info=None):
        return self.agent.act(state, clip(reward), info)

    def terminal(self, reward, info=None):
        return self.agent.terminal(clip(reward), info)

def clip(reward):
    return np.sign(reward)

class FrameStack(Body):
    def __init__(self, agent, size=4):
        super().__init__(agent)
        self._state = []
        self._size = 4

    def initial(self, state, info=None):
        self._state = [state] * self._size
        return self.agent.initial(stack(self._state), info)

    def act(self, state, reward, info=None):
        self._state = self._state[1:] + [state]
        return self.agent.act(stack(self._state), reward, info)

class FireOnReset(Body):
    def __init__(self, agent, env):
        super().__init__(agent)
        self._frames = 0

    def initial(self, state, info=None):
        self._frames = 1
        return torch.tensor([1])

    def act(self, state, reward, info=None):
        if self._frames == 1:
            self._frames += 1
            return torch.tensor([2])
        if self._frames == 2:
            self._frames += 1
            return self.agent.initial(state, info)
        return self.agent.act(state, reward, info)
        

class DeepmindAtariBodyInner(Body):
    def __init__(
            self,
            agent,
            env,
            frameskip=4,
    ):
        super().__init__(agent)
        self.env = env
        self.frameskip = frameskip
        self._action = None
        self._reward = 0
        self._info = None
        self._skipped_frames = 0
        self._lives = 0

    def initial(self, state, info=None):
        self._action = self.agent.initial(state, info)
        self._reward = 0
        self._info = info
        self._skipped_frames = 0
        self._lives = self._get_lives()
        return self._action

    def act(self, state, reward, info=None):
        if self._lost_life():
            self.terminal(self._reward, self._info)
            return self.initial(state, info)

        self._update_state(state, reward, info)
        if self._should_choose_action():
            self._choose_action(state)
        return self._action

    def terminal(self, reward, info=None):
        self._reward += reward
        self._info = info
        return self.agent.terminal(self._reward, self._info)

    def _update_state(self, state, reward, info):
        self._reward += reward
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

    def _choose_action(self, state):
        self._action = self.agent.act(
            state,
            self._reward,
            self._info
        )
        self._reward = 0
        self._skipped_frames = 0
        return self._action

class DeepmindAtariBody(Body):
    '''
    Enable the Agent to play Atari games DeepMind Style

    Implements the following features:
    1. Frame preprocessing (deflicker + downsample + grayscale)
    2. Frame stacking
    3. Action Repeat (TODO)
    4. Reward clipping (-1, 0, 1)
    5. Episodic lives
    6. Fire on reset
    7. No-op on reset
    '''

    def __init__(self, agent, env, frame_stack=4, deflicker=True, noop_max=30):
        agent = DeepmindAtariBodyInner(agent, env)
        agent = RewardClipping(agent)
        # pylint: disable=protected-access
        if env._env.unwrapped.get_action_meanings()[1] == 'FIRE':
            agent = FireOnReset(agent, env)
        agent = FrameStack(agent, size=frame_stack)
        agent = AtariVisionPreprocessor(agent, deflicker=deflicker)
        if noop_max > 0:
            agent = NoopBody(agent, noop_max)
        super().__init__(agent)
