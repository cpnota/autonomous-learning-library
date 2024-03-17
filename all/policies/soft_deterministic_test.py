import unittest

import numpy as np
import torch
import torch_testing as tt
from gymnasium.spaces import Box

from all import nn
from all.approximation import DummyCheckpointer
from all.core import State
from all.policies import SoftDeterministicPolicy

STATE_DIM = 2
ACTION_DIM = 3


class TestSoftDeterministic(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = nn.Sequential(nn.Linear0(STATE_DIM, ACTION_DIM * 2))
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.01)
        self.space = Box(np.array([-1, -1, -1]), np.array([1, 1, 1]))
        self.policy = SoftDeterministicPolicy(
            self.model, self.optimizer, self.space, checkpointer=DummyCheckpointer()
        )

    def test_output_shape(self):
        state = State(torch.randn(1, STATE_DIM))
        action, log_prob = self.policy(state)
        self.assertEqual(action.shape, (1, ACTION_DIM))
        self.assertEqual(log_prob.shape, torch.Size([1]))

        state = State(torch.randn(5, STATE_DIM))
        action, log_prob = self.policy(state)
        self.assertEqual(action.shape, (5, ACTION_DIM))
        self.assertEqual(log_prob.shape, torch.Size([5]))

    def test_step_one(self):
        state = State(torch.randn(1, STATE_DIM))
        self.policy(state)
        self.policy.step()

    def test_converge(self):
        state = State(torch.randn(1, STATE_DIM))
        target = torch.tensor([0.25, 0.5, -0.5])

        for _ in range(0, 200):
            action, _ = self.policy(state)
            loss = ((target - action) ** 2).mean()
            loss.backward()
            self.policy.step()

        self.assertLess(loss, 0.2)

    def test_scaling(self):
        torch.manual_seed(0)
        state = State(torch.randn(1, STATE_DIM))
        policy1 = SoftDeterministicPolicy(
            self.model,
            self.optimizer,
            Box(np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])),
        )
        action1, log_prob1 = policy1(state)

        # reset seed and sample same thing, but with different scaling
        torch.manual_seed(0)
        state = State(torch.randn(1, STATE_DIM))
        policy2 = SoftDeterministicPolicy(
            self.model,
            self.optimizer,
            Box(np.array([-2.0, -1.0, -1.0]), np.array([2.0, 1.0, 1.0])),
        )
        action2, log_prob2 = policy2(state)

        # check scaling was correct
        tt.assert_allclose(action1 * torch.tensor([2, 1, 1]), action2)
        tt.assert_allclose(log_prob1 - np.log(2), log_prob2)


if __name__ == "__main__":
    unittest.main()
