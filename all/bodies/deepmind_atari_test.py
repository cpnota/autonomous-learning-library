import unittest
import torch
import torch_testing as tt
from all.agents import Agent
from all.environments import GymEnvironment
from all.bodies.deepmind_atari import DeepmindAtariBody

class MockAgent(Agent):
    def __init__(self):
        self.state = None
        self.reward = None
        self.info = None

    def initial(self, state, info=None):
        self.state = state
        self.info = info
        return torch.tensor([0])

    def act(self, state, reward, info=None):
        self.state = state
        self.reward = reward
        self.info = info
        return torch.tensor([0])

    def terminal(self, reward, info=None):
        self.reward = reward
        self.info = info

class DeepmindAtariBodyTest(unittest.TestCase):
    def setUp(self):
        self.agent = MockAgent()
        self.env = GymEnvironment('PongNoFrameskip-v4')
        self.body = DeepmindAtariBody(self.agent, self.env)

    def test_initial_state(self):
        self.env.reset()
        action = self.body.initial(self.env.state)
        tt.assert_equal(action, torch.tensor([0]))
        self.assertEqual(self.agent.state.shape, (1, 4, 105, 80))

    def test_second_state(self):
        self.env.reset()
        self.env.step(self.body.initial(self.env.state))
        action = self.body.act(self.env.state, self.env.reward)
        tt.assert_equal(action, torch.tensor([0]))
        self.assertEqual(self.agent.state.shape, (1, 4, 105, 80))

    def test_several_steps(self):
        self.env.reset()
        self.env.step(self.body.initial(self.env.state))
        for _ in range(10):
            reward = -5 # should be clipped
            action = self.body.act(self.env.state, reward)
            self.env.step(action)
        tt.assert_equal(action, torch.tensor([0]))
        self.assertEqual(self.agent.state.shape, (1, 4, 105, 80))
        self.assertEqual(self.agent.reward, -4)

    def test_terminal_state(self):
        self.env.reset()
        self.env.step(self.body.initial(self.env.state))
        for _ in range(12):
            reward = -5 # should be clipped
            action = self.body.act(self.env.state, reward)
            self.env.step(action)
        self.body.terminal(-1)
        tt.assert_equal(action, torch.tensor([0]))
        self.assertEqual(self.agent.state.shape, (1, 4, 105, 80))
        self.assertEqual(self.agent.reward, -2)

if __name__ == '__main__':
    unittest.main()
