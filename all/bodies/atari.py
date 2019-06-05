import torch
import numpy as np
from all.environments import State
from .abstract import Body
from .parallel import ParallelBody, ParallelRepeatActions

# pylint: disable=protected-access
NOOP_ACTION = torch.tensor([0])

class NoopBody(Body):
    def __init__(self, agent, noop_max):
        super().__init__(agent)
        self.noop_max = noop_max
        self.noops = 0
        self.actions_taken = 0

    def initial(self, state):
        self.noops = np.random.randint(self.noop_max) + 1
        self.actions_taken = 0
        return NOOP_ACTION

    def act(self, state, reward):
        self.actions_taken += 1
        if self.actions_taken < self.noops:
            return NOOP_ACTION
        if self.actions_taken == self.noops:
            return self.agent.initial(state)
        return self.agent.act(state, reward)

    def terminal(self, state, reward):
        if self.actions_taken >= self.noops:
            self.agent.terminal(state, reward)
        # otherwise, the poor agent never stood a chance

class Deflicker(Body):
    _previous_frame = None

    def initial(self, state):
        self._previous_frame = state.raw
        return self.agent.initial(state)

    def act(self, state, reward):
        frame = state.raw
        frame, self._previous_frame = torch.max(
            frame, self._previous_frame), frame
        deflickered = State(frame, state.mask, state.info)
        return self.agent.act(deflickered, reward)

class AtariVisionPreprocessor(Body):
    def initial(self, state):
        return self.agent.initial(State(
            self._preprocess(state.raw),
            state.mask,
            state.info
        ))

    def act(self, state, reward):
        state = State(
            self._preprocess(state.raw),
            state.mask,
            state.info
        )
        return self.agent.act(state, reward)

    def terminal(self, state, reward):
        state = State(
            self._preprocess(state.raw),
            state.mask,
            state.info
        )
        return self.agent.terminal(state, reward)

    def _preprocess(self, frame):
        return to_grayscale(downsample(frame)).unsqueeze(1)

def to_grayscale(frame):
    return torch.mean(frame.float(), dim=3).byte()

def downsample(frame):
    return frame[:, ::2, ::2]

class FrameStack(Body):
    def __init__(self, agent, size=4):
        super().__init__(agent)
        self._frames = []
        self._size = size

    def initial(self, state):
        self._frames = [state.raw] * self._size
        return self.agent.initial(State(
            stack(self._frames),
            state.mask,
            state.info
        ))

    def act(self, state, reward):
        self._frames = self._frames[1:] + [state.raw]
        return self.agent.act(
            State(
                stack(self._frames),
                state.mask,
                state.info
            ),
            reward
        )

    def terminal(self, state, reward):
        self._frames = self._frames[1:] + [state.raw]
        return self.agent.act(
            State(
                stack(self._frames),
                state.mask,
                state.info
            ),
            reward
        )

def stack(frames):
    return torch.cat(frames, dim=1)

class EpisodicLives(Body):
    def __init__(self, agent, env):
        super().__init__(agent)
        self._env = env
        self._lives = 0

    def initial(self, state):
        self._lives = self._get_lives()
        return self.agent.initial(state)

    def act(self, state, reward):
        if self._lost_life():
            state = State(state.raw, state.mask * 0, state.info)
            self.terminal(state, reward)
            self._lives = self._get_lives()
            return self.initial(state)
        return self.agent.act(state, reward)

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

    def initial(self, state):
        self._frames = 1
        return torch.tensor([1])

    def act(self, state, reward):
        if self._frames == 1:
            self._frames += 1
            return torch.tensor([2])
        if self._frames == 2:
            self._frames += 1
            return self.agent.initial(state)
        return self.agent.act(state, reward)

class RewardClipping(Body):
    def act(self, state, reward):
        return self.agent.act(state, np.sign(reward))

    def terminal(self, state, reward):
        return self.agent.terminal(state, np.sign(reward))

class RepeatActions(Body):
    def __init__(self, agent, repeats=4):
        super().__init__(agent)
        self._repeats = repeats
        self._count = 0
        self._action = None
        self._reward = 0

    def initial(self, state):
        self._action = self.agent.initial(state)
        self._reward = 0
        self._count = 0
        return self._action

    def act(self, state, reward):
        self._count += 1
        self._reward += reward
        if self._count == self._repeats:
            self._action = self.agent.act(state, self._reward)
            self._reward = 0
            self._count = 0
        return self._action

    def terminal(self, state, reward):
        self._reward += reward
        return self.agent.terminal(state, self._reward)

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


class ParallelAtariBody(Body):
    '''Parallel version of DeepmindatariBody'''
    def __init__(
            self,
            agent,
            envs,
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
            agent = ParallelRepeatActions(agent, repeats=action_repeat)

        def make_body(agent, env):
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
            return agent

        agent = ParallelBody(agent, envs, make_body)

        super().__init__(agent)
