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
        self._size = size

    def initial(self, state, info=None):
        self._state = [state] * self._size
        return self.agent.initial(stack(self._state), info)

    def act(self, state, reward, info=None):
        self._state = self._state[1:] + [state]
        return self.agent.act(stack(self._state), reward, info)

class FireOnReset(Body):
    def __init__(self, agent):
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

class EpisodicLives(Body):
    def __init__(
        self,
        agent,
        env
    ):
        super().__init__(agent)
        self._env = env
        self._lives = 0

    def initial(self, state, info=None):
        self._lives = self._get_lives()
        return self.agent.initial(state, info)

    def act(self, state, reward, info=None):
        if self._lost_life():
            self.terminal(reward, info)
            self._lives = self._get_lives()
            return self.initial(state, info)
        return self.agent.act(state, reward, info)

    def _lost_life(self):
        lives = self._get_lives()
        return lives < self._lives and lives > 0

    def _get_lives(self):
        # pylint: disable=protected-access
        return self._env._env.unwrapped.ale.lives()

class RepeatActions(Body):
    def __init__(
        self,
        agent,
        repeats=4
    ):
        super().__init__(agent)
        self._repeats = repeats
        self._count = 0
        self._action = None
        self._reward = 0

    def initial(self, state, info=None):
        self._action = self.agent.initial(state, info)
        self._reward = 0
        self._count = 0
        return self._action

    def act(self, state, reward, info=None):
        self._count += 1
        self._reward += reward
        if self._count == self._repeats:
            self._action = self.agent.act(state, self._reward, info)
            self._reward = 0
            self._count = 0
        return self._action

    def terminal(self, reward, info=None):
        self._reward += reward
        return self.agent.terminal(self._reward, info)

class DeepmindAtariBody(Body):
    '''
    Enable the Agent to play Atari games DeepMind Style

    Implements the following features:
    1. Frame preprocessing (deflicker + downsample + grayscale)
    2. Frame stacking
    3. Action Repeat
    4. Reward clipping
    5. Episodic lives
    6. Fire on reset
    7. No-op on reset
    '''
    def __init__(self, agent, env, action_repeat=4, frame_stack=4, deflicker=True, noop_max=30):
        agent = RepeatActions(agent, repeats=action_repeat)
        agent = RewardClipping(agent)
        # pylint: disable=protected-access
        if env._env.unwrapped.get_action_meanings()[1] == 'FIRE':
            agent = FireOnReset(agent)
        agent = EpisodicLives(agent, env)
        agent = FrameStack(agent, size=frame_stack)
        agent = AtariVisionPreprocessor(agent, deflicker=deflicker)
        if noop_max > 0:
            agent = NoopBody(agent, noop_max)
        super().__init__(agent)
