import unittest
import torch
from torch import nn
from all.environments import State
from all.policies import GaussianPolicy

STATE_DIM = 2
ACTION_DIM = 3

class TestSoftmax(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = nn.Sequential(
            nn.Linear(STATE_DIM, ACTION_DIM * 2)
        )
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.01)
        self.policy = GaussianPolicy(self.model, optimizer, ACTION_DIM)

    def test_output_shape(self):
        state = State(torch.randn(1, STATE_DIM))
        action = self.policy(state)
        self.assertEqual(action.shape, (1, ACTION_DIM))
        state = State(torch.randn(5, STATE_DIM))
        action = self.policy(state)
        self.assertEqual(action.shape, (5, ACTION_DIM))

    def test_reinforce_one(self):
        state = State(torch.randn(1, STATE_DIM))
        self.policy(state)
        self.policy.reinforce(torch.tensor([1]).float())

    def test_converge(self):
        state = State(torch.randn(1, STATE_DIM))
        target = torch.tensor([1., 2., -1.])

        for _ in range(0, 1000):
            action = self.policy(state)
            loss = torch.abs(target - action).mean()
            self.policy.reinforce(-loss)

        self.assertTrue(loss < 1)

if __name__ == '__main__':
    unittest.main()
