import unittest
import torch
import torch_testing as tt
import numpy as np
from all.agents import Agent
from all.environments import AtariEnvironment
from all.bodies.parallel import ParallelRepeatActions, ParallelAtariBody

class MockAgent(Agent):
    def __init__(self, n, max_action=6):
        self._actions = torch.zeros(n).long()
        self._max_action = max_action
        self._n = n
        self._states = []
        self._rewards = []

    def act(self, state, reward, info=None):
        self._actions = (self._actions + 1) % self._max_action
        self._states.append(state)
        self._rewards.append(reward.view(1, -1))
        return self._actions

class ParallelRepeatActionsTest(unittest.TestCase):
    def test_repeat_actions(self):
        states = [
            ['state1', 'state2'],
            ['state3', None],
            ['state5', None],
            ['state7', 'state8'],
            ['state9', 'state10'],
            ['state11', 'state12'],
            ['state13', 'state14']
        ]
        rewards = torch.ones(2)

        agent = MockAgent(2)
        body = ParallelRepeatActions(agent, repeats=3)

        actions = body.act(states[0], rewards)
        self.assert_array_equal(actions, [1, 1])
        actions = body.act(states[1], rewards)
        self.assert_array_equal(actions, [1, None])
        actions = body.act(states[2], rewards)
        self.assert_array_equal(actions, [1, None])
        actions = body.act(states[3], rewards)
        self.assert_array_equal(actions, [2, 2])
        actions = body.act(states[4], rewards)
        self.assert_array_equal(actions, [2, 2])
        actions = body.act(states[5], rewards)
        self.assert_array_equal(actions, [2, 2])
        actions = body.act(states[6], rewards)
        self.assert_array_equal(actions, [3, 3])

        self.assert_array_equal(agent._states, [
            states[0],
            states[3],
            states[6]
        ])
        tt.assert_equal(torch.cat(agent._rewards), torch.tensor([
            [1, 1],
            [3, 3],
            [3, 3]
        ]))

    def assert_array_equal(self, actual, expected):
        for i, exp in enumerate(expected):
            self.assertEqual(actual[i], exp, msg=(("\nactual: %s\nexpected: %s") % (actual, expected)))

class ParallelAtariBodyTest(unittest.TestCase):
    def test_runs(self):
        np.random.seed(0)
        torch.random.manual_seed(0)
        n = 4
        envs = []
        for i in range(n):
            env = AtariEnvironment('Breakout')
            env.reset()
            envs.append(env)
        agent = MockAgent(n, max_action=4)
        body = ParallelAtariBody(agent, envs, noop_max=4)

        for t in range(200):
            states = [env.state for env in envs]
            rewards = torch.tensor([env.reward for env in envs]).float()
            actions = agent.act(states, rewards)
            for i, env in enumerate(envs):
                env.step(actions[i])

        tt.assert_equal(agent._states[199][0], agent._states[199][2])

if __name__ == '__main__':
    unittest.main()
