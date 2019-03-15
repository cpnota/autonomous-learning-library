import unittest
import torch
import torch_testing as tt
import numpy as np
from all.agents import Agent
from all.environments import GymEnvironment
from all.bodies.deepmind_atari import DeepmindAtariBody

NOOP_ACTION = torch.tensor([0])
INITIAL_ACTION = torch.tensor([3])
ACT_ACTION = torch.tensor([4])

class MockAgent(Agent):
    def __init__(self):
        self.state = None
        self.reward = None
        self.info = None

    def initial(self, state, info=None):
        self.state = state
        self.info = info
        return INITIAL_ACTION

    def act(self, state, reward, info=None):
        self.state = state
        self.reward = reward
        self.info = info
        return ACT_ACTION

    def terminal(self, reward, info=None):
        self.reward = reward
        self.info = info

class ALE():
    def lives(self):
        return 1

class Unwrapped():
    ale = ALE()
    def get_action_meanings(self):
        return ['TEST', 'ACTIONS']

class InnerEnv():
    unwrapped = Unwrapped()

class MockEnv():
    _env = InnerEnv()

class DeepmindAtariBodyTest(unittest.TestCase):
    def setUp(self):
        self.agent = MockAgent()
        self.env = MockEnv()
        self.body = DeepmindAtariBody(self.agent, self.env, noop_max=0)

    def test_initial_state(self):
        frame = torch.ones((1, 3, 4, 4))
        action = self.body.initial(frame)
        tt.assert_equal(action, INITIAL_ACTION)
        tt.assert_equal(self.agent.state, torch.ones(1, 4, 2, 2))

    def test_deflicker(self):
        frame1 = torch.ones((1, 3, 4, 4))
        frame2 = torch.ones((1, 3, 4, 4))
        frame3 = torch.ones((1, 3, 4, 4)) * 2
        self.body.initial(frame1)
        self.body.act(frame2, 0)
        self.body.act(frame3, 0)
        self.body.act(frame2, 0)
        self.body.act(frame2, 0)
        expected = torch.cat((
            torch.ones(1, 2, 2),
            torch.ones(2, 2, 2) * 2,
            torch.ones(1, 2, 2)
        )).unsqueeze(0)
        tt.assert_equal(self.agent.state, expected)

class DeepmindAtariBodyPongTest(unittest.TestCase):
    def setUp(self):
        self.agent = MockAgent()
        self.env = GymEnvironment('PongNoFrameskip-v4')
        self.body = DeepmindAtariBody(self.agent, self.env, noop_max=0)

    def test_initial_state(self):
        self.env.reset()
        action = self.body.initial(self.env.state)
        tt.assert_equal(action, torch.tensor([1])) # fire on reset 1

    def test_second_state(self):
        self.env.reset()
        self.env.step(self.body.initial(self.env.state))
        action = self.body.act(self.env.state, self.env.reward)
        tt.assert_equal(action, torch.tensor([2])) # fire on reset 2

    def test_several_steps(self):
        self.env.reset()
        self.env.step(self.body.initial(self.env.state))
        self.env.step(self.body.act(self.env.state, -5))
        for _ in range(4):
            action = self.body.act(self.env.state, -5)
            self.assertEqual(self.agent.state.shape, (1, 4, 105, 80))
            tt.assert_equal(action, INITIAL_ACTION)
            self.env.step(action)
        for _ in range(10):
            reward = -5  # should be clipped
            self.assertEqual(self.agent.state.shape, (1, 4, 105, 80))
            action = self.body.act(self.env.state, reward)
            tt.assert_equal(action, ACT_ACTION)
            self.env.step(action)
        self.assertEqual(self.agent.reward, -4)

    def test_terminal_state(self):
        self.env.reset()
        self.env.step(self.body.initial(self.env.state))
        for _ in range(11):
            reward = -5  # should be clipped
            action = self.body.act(self.env.state, reward)
            self.env.step(action)
        self.body.terminal(-1)
        tt.assert_equal(action, ACT_ACTION)
        self.assertEqual(self.agent.state.shape, (1, 4, 105, 80))
        self.assertEqual(self.agent.reward, -2)

class NoopTest(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.agent = MockAgent()
        self.env = MockEnv()
        self.frame = torch.ones((1, 3, 4, 4))
        self.body = DeepmindAtariBody(self.agent, self.env, noop_max=10)

    def test_noops(self):
        action = self.body.initial(self.frame)
        tt.assert_equal(action, torch.tensor([0]))
        for _ in range(4):
            action = self.body.act(self.frame, 0)
            tt.assert_equal(action, torch.tensor([0]))
        for _ in range(4):
            action = self.body.act(self.frame, 0)
            tt.assert_equal(action, INITIAL_ACTION)
        for _ in range(4):
            action = self.body.act(self.frame, 0)
            tt.assert_equal(action, ACT_ACTION)

if __name__ == '__main__':
    unittest.main()
