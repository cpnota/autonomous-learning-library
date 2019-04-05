import torch
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
            self._rewards = rewards
        else:
            self._rewards = self._rewards + rewards
        self._count += 1
        if self._count >= self._repeats:
            self._actions = self.agent.act(states, self._rewards, info)
            self._rewards = self._rewards * 0
            self._count = 0
        else:
            for i, _ in enumerate(states):
                if states[i] is None:
                    self._actions[i] = None
        return self._actions

class Joiner(Body):
    '''Internal class used by Parallelizer'''
    def __init__(self, parallelizer):
        super().__init__(parallelizer)

    def act(self, state, reward, info=None):
        self.agent.join(state, reward, info)

class ParallelBody(Body):
    '''Allows the use of single-environment Body code with multi-environment Agents'''
    def __init__(self, agent, make_body, n):
        super().__init__(agent)
        self._n = n
        self._i = 0
        self._states = [None] * n
        self._rewards = None
        self._info = [None] * n
        self._bodies = [None] * n
        self._ready = [False] * n
        self._actions = [None] * n
        for i in range(n):
            joiner = Joiner(self)
            self._bodies[i] = make_body(joiner)
            self._actions[i] = None

    def act(self, states, rewards, infos):
        if self._rewards is None:
            self._rewards = rewards.clone()
        actions = [None] * self._n
        ready = True
        for i in range(self._n):
            self._i = i
            if not self._ready[i]:
                self._rewards[i] = rewards[i]
                action = self._bodies[i].act(states[i], rewards[i], infos[i])
                if action is None:
                    self._ready[i] = True
                else:
                    ready = False
                actions[i] = action
        if ready:
            actions = self.agent.act(self._states, self._rewards, self._info)
            self._ready = [False] * self._n
        return actions

    def join(self, state, reward, info):
        self._states[self._i] = state
        self._rewards[self._i] = reward
        self._info[self._i] = info
