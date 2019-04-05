import unittest
import torch
import torch_testing as tt
import numpy as np
from all.agents import Agent
from all.environments import GymEnvironment
from all.bodies import Body, ParallelBody
from all.bodies.parallel import ParallelBody

class MockAgent(Agent):
    def __init__(self, n):
        self._actions = 0
        self._n = n

    def act(self, state, reward, info=None):
        self._actions += 1
        return [self._actions] * self._n

class RepeatBody(Body):
    def __init__(self, agent):
        super().__init__(agent)
        self._repeat = np.random.randint(1, 4)
        self._i = -1
        self._action = None

    def act(self, state, reward, info=None):
        if self._repeat == 1:
            if self._i == 1:
                return None
            self._action = self.agent.act(state, reward, info)
            if self._action is None:
                self._i = 1
            return self._action

        if self._i >= self._repeat or self._i == -1:
            self._action = self.agent.act(state, reward, info)
            self._i = 1
            return self._action

        if self._action is None:
            self._action = self.agent.act(state, reward, info)
        self._i += 1
        return self._action

class ParallelBodyTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_parallel_four(self):
        agent = MockAgent(4)
        agent = ParallelBody(agent, RepeatBody, 4)

        states = ['state'] * 4
        rewards = [0] * 4
        infos = [None] * 4

        for t in range(10):
            action = agent.act(states, rewards, infos)
            print(t, action)

if __name__ == '__main__':
    unittest.main()
