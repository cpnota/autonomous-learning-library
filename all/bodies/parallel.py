import torch
from .abstract import Body

class Joiner(Body):
    '''Internal class used by Parallelizer'''
    def __init__(self, parallelizer):
        super().__init__(parallelizer)
        self.parallelizer = parallelizer
        self._actions = [None] * parallelizer._n

    def act(self, state, reward, info=None):
        self.parallelizer.join(state, reward, info)
        # print('hi', self.parallelizer._actions)
        return self.parallelizer._actions[self.parallelizer._i]


class ParallelBody(Body):
    '''Allows the use of single-environment Body code with multi-environment Agents'''
    def __init__(self, agent, make_body, n):
        super().__init__(agent)
        self._n = n
        self._i = 0
        self._states = [None] * n
        self._rewards = [None] * n
        self._info = [None] * n
        self._bodies = [None] * n
        self._ready = [False] * n
        self._actions = [None] * n
        for i in range(n):
            joiner = Joiner(self)
            self._bodies[i] = make_body(joiner)
            self._actions[i] = None

    def act(self, states, rewards, infos):
        actions = [None] * self._n
        ready = True
        for i in range(self._n):
            self._i = i
            if not self._ready[i]:
                action = self._bodies[i].act(states[i], rewards[i], infos[i])
                if action is None:
                    self._ready[i] = True
                    actions[i] = None
                else:
                    self._ready[i] = False
                    ready = False
                actions[i] = action
        # print('actions', actions)

        if ready:
            # print('ready')
            actions = self.agent.act(self._states, self._rewards, self._info)
            # print('ready actions', actions)
            self._actions = actions
            self._ready = [False] * self._n
        else:
            self._actions = [None] * self._n
        return actions

    def join(self, state, reward, info):
        self._states[self._i] = state
        self._rewards[self._i] = reward
        self._info[self._i] = info
