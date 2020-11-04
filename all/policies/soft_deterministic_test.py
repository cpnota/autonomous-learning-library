import unittest
import torch
import numpy as np
import torch_testing as tt
from gym.spaces import Box
from all import nn
from all.approximation import DummyCheckpointer
from all.core import State
from all.policies import SoftDeterministicPolicy

STATE_DIM = 2
ACTION_DIM = 3


class TestSoftDeterministic(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = nn.Sequential(
            nn.Linear0(STATE_DIM, ACTION_DIM * 2)
        )
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.01)
        self.space = Box(np.array([-1, -1, -1]), np.array([1, 1, 1]))
        self.policy = SoftDeterministicPolicy(
            self.model,
            self.optimizer,
            self.space,
            checkpointer=DummyCheckpointer()
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
        self.space = Box(np.array([-10, -5, 100]), np.array([10, -2, 200]))
        self.policy = SoftDeterministicPolicy(
            self.model,
            self.optimizer,
            self.space
        )
        state = State(torch.randn(1, STATE_DIM))
        action, log_prob = self.policy(state)
        tt.assert_allclose(action, torch.tensor([[-3.09055, -4.752777, 188.98222]]))
        tt.assert_allclose(log_prob, torch.tensor([-0.397002]), rtol=1e-4)


if __name__ == '__main__':
    unittest.main()
