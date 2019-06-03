from all.environments import State
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
                if not states[i].mask:
                    self._actions[i] = None
        return self._actions


class Joiner(Body):
    '''Internal class used by Parallelizer'''

    def initial(self, state):
        self.agent.join(state, 0)

    def act(self, state, reward):
        self.agent.join(state, reward)

    def terminal(self, state, reward):
        self.agent.join(state, reward)


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
        self._bodies = [None] * n
        self._ready = [False] * n
        self._actions = [None] * n
        for i in range(n):
            joiner = Joiner(self)
            self._bodies[i] = make_body(joiner, envs[i])
            self._actions[i] = None

    def act(self, states, rewards):
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
                    action = self._bodies[i].initial(states[i])
                elif states[i] is None:
                    self._bodies[i].terminal(states[i], rewards[i].item())
                else:
                    action = self._bodies[i].act(
                        states[i], rewards[i].item())

                if action is None:
                    self._ready[i] = True
                else:
                    ready = False
                actions[i] = action
        self._last_states = states
        if ready:
            states = State.from_list(self._states)
            actions = self.agent.act(states, self._rewards)
            self._ready = [False] * self._n
            self._states = [None] * self._n
        return actions

    def join(self, state, reward):
        self._states[self._i] = state
        self._rewards[self._i] = reward
