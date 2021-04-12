import unittest
import numpy as np
import torch
from torch import nn
import torch_testing as tt
from gym.spaces import Box
from all.approximation import DummyCheckpointer
from all.core import State
from all.policies import GaussianPolicy

STATE_DIM = 2
ACTION_DIM = 3


class TestGaussian(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.space = Box(np.array([-1, -1, -1]), np.array([1, 1, 1]))
        self.model = nn.Sequential(
            nn.Linear(STATE_DIM, ACTION_DIM * 2)
        )
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.01)
        self.policy = GaussianPolicy(self.model, optimizer, self.space, checkpointer=DummyCheckpointer())

    def test_output_shape(self):
        state = State(torch.randn(1, STATE_DIM))
        action = self.policy(state).sample()
        self.assertEqual(action.shape, (1, ACTION_DIM))
        state = State(torch.randn(5, STATE_DIM))
        action = self.policy(state).sample()
        self.assertEqual(action.shape, (5, ACTION_DIM))

    def test_reinforce_one(self):
        state = State(torch.randn(1, STATE_DIM))
        dist = self.policy(state)
        action = dist.sample()
        log_prob1 = dist.log_prob(action)
        loss = -log_prob1.mean()
        self.policy.reinforce(loss)

        dist = self.policy(state)
        log_prob2 = dist.log_prob(action)

        self.assertGreater(log_prob2.item(), log_prob1.item())

    def test_converge(self):
        state = State(torch.randn(1, STATE_DIM))
        target = torch.tensor([1., 2., -1.])

        for _ in range(0, 1000):
            dist = self.policy(state)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            error = ((target - action) ** 2).mean()
            loss = (error * log_prob).mean()
            self.policy.reinforce(loss)

        self.assertTrue(error < 1)

    def test_eval(self):
        state = State(torch.randn(1, STATE_DIM))
        dist = self.policy.no_grad(state)
        tt.assert_almost_equal(dist.mean, torch.tensor([[-0.237, 0.497, -0.058]]), decimal=3)
        tt.assert_almost_equal(dist.entropy(), torch.tensor([4.254]), decimal=3)
        best = self.policy.eval(state).sample()
        tt.assert_almost_equal(best, torch.tensor([[-0.888, -0.887, 0.404]]), decimal=3)


if __name__ == '__main__':
    unittest.main()
