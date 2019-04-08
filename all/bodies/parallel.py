import torch
from .atari import *
from .abstract import Body

class ParallelRepeatActions(Body):
    def __init__(self, agent, repeats=4):
        super().__init__(agent)
        self._repeats = repeats
        self._count = repeats
        self._actions = None
        self._rewards = None

    def act(self, states, rewards, info=None):
        if self._rewards is None:
            self._rewards = rewards.clone()
        else:
            self._rewards += rewards
        self._count += 1
        if self._count >= self._repeats:
            self._actions = list(self.agent.act(states, self._rewards, info))
            self._rewards = self._rewards * 0
            self._count = 0
        else:
            for i, _ in enumerate(states):
                if states[i] is None:
                    self._actions[i] = None
        return self._actions

class Joiner(Body):
    '''Internal class used by Parallelizer'''
    def init(self, state, info=None):
        self.agent.join(state, 0, info)

    def act(self, state, reward, info=None):
        self.agent.join(state, reward, info)

    def terminal(self, reward, info=None):
        self.agent.join(None, reward, info)

class ParallelBody(Body):
    '''Allows the use of single-environment Body code with multi-environment Agents'''
    def __init__(self, agent, envs, make_body):
        super().__init__(agent)
        n = len(envs)
        self._n = n
        self._i = 0
        self._states = [None] * n
        self._last_states = [None] * n
        self._rewards = None
        self._info = [None] * n
        self._bodies = [None] * n
        self._ready = [False] * n
        self._actions = [None] * n
        for i in range(n):
            joiner = Joiner(self)
            self._bodies[i] = make_body(joiner, envs[i])
            self._actions[i] = None

    def act(self, states, rewards, infos):
        if infos is None:
            infos = [None] * self._n
        if self._rewards is None:
            self._rewards = rewards.clone()
        actions = [None] * self._n
        ready = True
        for i in range(self._n):
            self._i = i
            if not self._ready[i]:
                self._rewards[i] = rewards[i]
                
                action = None
                if self._last_states[i] is None:
                    action = self._bodies[i].initial(states[i], infos[i])
                elif states[i] is None:
                    self._bodies[i].terminal(rewards[i], infos[i])
                else:
                    action = self._bodies[i].act(states[i], rewards[i], infos[i])

                if action is None:
                    self._ready[i] = True
                else:
                    ready = False
                actions[i] = action
        if ready:
            actions = self.agent.act(self._states, self._rewards, self._info)
            self._ready = [False] * self._n
            self._states = [None] * self._n
        self._last_states = states
        return actions

    def join(self, state, reward, info):
        self._states[self._i] = state
        self._rewards[self._i] = reward
        self._info[self._i] = info

class ParallelAtariBody(Body):
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
