import unittest
import torch
import torch_testing as tt
import numpy as np
from gym.spaces import Box
from all import nn
from all.approximation import FixedTarget
from all.environments import State
from all.policies import DeterministicPolicy

STATE_DIM = 2
ACTION_DIM = 3

class TestDeterministic(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = nn.Sequential(
            nn.Linear0(STATE_DIM, ACTION_DIM)
        )
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.01)
        self.space = Box(np.array([-1, -1, -1]), np.array([1, 1, 1]))
        self.policy = DeterministicPolicy(
            self.model,
            self.optimizer,
            self.space,
            0.5
        )

    def test_output_shape(self):
        state = State(torch.randn(1, STATE_DIM))
        action = self.policy(state)
        self.assertEqual(action.shape, (1, ACTION_DIM))
        state = State(torch.randn(5, STATE_DIM))
        action = self.policy(state)
        self.assertEqual(action.shape, (5, ACTION_DIM))

    def test_clipping(self):
        space = Box(np.array([-0.1, -0.1, -0.1]), np.array([0.1, 0.1, 0.1]))
        self.policy = DeterministicPolicy(
            self.model,
            self.optimizer,
            space,
            0.5
        )
        state = State(torch.randn(1, STATE_DIM))
        action = self.policy(state).detach().numpy()
        np.testing.assert_array_almost_equal(action, np.array([[-0.1, -0.1, 0.1]]))

    def test_step_one(self):
        state = State(torch.randn(1, STATE_DIM))
        self.policy(state)
        self.policy.step()

    def test_converge(self):
        state = State(torch.randn(1, STATE_DIM))
        target = torch.tensor([1., 2., -1.])

        for _ in range(0, 100):
            action = self.policy.greedy(state)
            loss = torch.abs(target - action).mean()
            loss.backward()
            self.policy.step()

        self.assertTrue(loss < 0.1)

    def test_target(self):
        self.policy = DeterministicPolicy(
            self.model,
            self.optimizer,
            self.space,
            0.5,
            target=FixedTarget(3)
        )

        # choose initial action
        state = State(torch.ones(1, STATE_DIM))
        action = self.policy.greedy(state)
        tt.assert_equal(action, torch.zeros(1, ACTION_DIM))

        # run update step, make sure target network doesn't change
        action.sum().backward(retain_graph=True)
        self.policy.step()
        tt.assert_equal(self.policy.eval(state), torch.zeros(1, ACTION_DIM))

        # again...
        action.sum().backward(retain_graph=True)
        self.policy.step()
        tt.assert_equal(self.policy.eval(state), torch.zeros(1, ACTION_DIM))

        # third time, target should be updated
        action.sum().backward(retain_graph=True)
        self.policy.step()
        tt.assert_allclose(
            self.policy.eval(state),
            torch.tensor([[-0.686739, -0.686739, -0.686739]])
        )

if __name__ == '__main__':
    unittest.main()
