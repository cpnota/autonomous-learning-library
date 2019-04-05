import unittest
import torch
import torch_testing as tt
import numpy as np
from all.agents import Agent
from all.environments import GymEnvironment
from all.bodies import Body, ParallelBody
from all.bodies.parallel import ParallelBody, ParallelRepeatActions

class MockAgent(Agent):
    def __init__(self, n):
        self._actions = 0
        self._n = n

    def act(self, state, reward, info=None):
        self._actions += 1
        return [self._actions] * self._n

class ParallelBodyTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_parallel_four(self):
        agent = MockAgent(4)
        agent = ParallelRepeatActions(agent)
        agent = ParallelBody(agent, Body, 4)

        states = ['state'] * 4
        rewards = torch.zeros(4)
        infos = [None] * 4

        for t in range(10):
            action = agent.act(states, rewards, infos)
            print(t, action)

if __name__ == '__main__':
    unittest.main()
