import torch
import numpy as np
from .abstract import Body

# pylint: disable=protected-access
NOOP_ACTION = torch.tensor([0])

class NoopBody(Body):
    def __init__(self, agent, noop_max):
        super().__init__(agent)
        self.noop_max = noop_max
        self.noops = 0
        self.actions_taken = 0

    def initial(self, state, info=None):
        self.noops = np.random.randint(self.noop_max) + 1
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
            self.agent.terminal(reward, info)
        # otherwise, the poor agent never stood a chance

class Deflicker(Body):
    _previous_frame = None

    def initial(self, frame, info=None):
        self._previous_frame = frame
        return self.agent.initial(frame, info)

    def act(self, frame, reward, info=None):
        frame, self._previous_frame = torch.max(
            frame, self._previous_frame), frame
        return self.agent.act(frame, reward, info)

class AtariVisionPreprocessor(Body):
    def initial(self, frame, info=None):
        return self.agent.initial(self._preprocess(frame), info)

    def act(self, frame, reward, info=None):
        return self.agent.act(self._preprocess(frame), reward, info)

    def _preprocess(self, frame):
        return to_grayscale(downsample(frame)).unsqueeze(1)

def to_grayscale(frame):
    return torch.mean(frame.float(), dim=3).byte()

def downsample(frame):
    return frame[:, ::2, ::2]

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

def stack(frames):
    return torch.cat(frames, dim=1)

class EpisodicLives(Body):
    def __init__(self, agent, env):
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
        return 0 < lives < self._lives

    def _get_lives(self):
        # pylint: disable=protected-access
        return self._env._env.unwrapped.ale.lives()

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

class RewardClipping(Body):
    def act(self, state, reward, info=None):
        return self.agent.act(state, np.sign(reward), info)

    def terminal(self, reward, info=None):
        return self.agent.terminal(np.sign(reward), info)

class RepeatActions(Body):
    def __init__(self, agent, repeats=4):
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

    Performs the following transformations in order:
    1. No-op on environment reset (random-ish starts)
    2. Frame preprocessing (deflicker + downsample + grayscale)
    3. Frame stacking (to perceive motion)
    4. Episodic lives (treat each life as an episode to shorten horizon)
    5. Fire on reset (Press the fire button to start on some games)
    6. Reward clipping (All are -1, 0, or 1)
    7. Action repeat (Repeat each chosen actions for four frames)
    '''

    def __init__(
            self,
            agent,
            env,
            action_repeat=4,
            clip_rewards=True,
            deflicker=True,
            episodic_lives=True,
            fire_on_reset=True,
            frame_stack=4,
            noop_max=30,
            preprocess=True,
    ):
        if action_repeat > 1:
            agent = RepeatActions(agent, repeats=action_repeat)
        if clip_rewards:
            agent = RewardClipping(agent)
        if fire_on_reset and env._env.unwrapped.get_action_meanings()[1] == 'FIRE':
            agent = FireOnReset(agent)
        if episodic_lives:
            agent = EpisodicLives(agent, env)
        if frame_stack > 1:
            agent = FrameStack(agent, size=frame_stack)
        if preprocess:
            agent = AtariVisionPreprocessor(agent)
        if deflicker:
            agent = Deflicker(agent)
        if noop_max > 0:
            agent = NoopBody(agent, noop_max)
        super().__init__(agent)
