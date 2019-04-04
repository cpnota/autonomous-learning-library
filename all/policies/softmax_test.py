import unittest
import torch
from torch import nn
import torch_testing as tt
from all.policies import SoftmaxPolicy

STATE_DIM = 2
ACTIONS = 3

class TestSoftmax(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = nn.Sequential(
            nn.Linear(STATE_DIM, ACTIONS)
        )
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.policy = SoftmaxPolicy(self.model, optimizer, ACTIONS)

    def test_run(self):
        state = torch.randn(1, STATE_DIM)
        action = self.policy(state)
        self.assertEqual(action.item(), 0)
        state = torch.randn(1, STATE_DIM)
        action = self.policy(state)
        self.assertEqual(action.item(), 2)
        self.policy.reinforce(torch.tensor([-1, 1000000]).float())
        action = self.policy(state)
        self.assertEqual(action.item(), 2)

    def test_multi_action(self):
        states = torch.randn(3, STATE_DIM)
        actions = self.policy(states)
        tt.assert_equal(actions, torch.tensor([2, 2, 0]))
        self.policy.reinforce(torch.tensor([[1, 2, 3]]).float())

    def test_list(self):
        torch.manual_seed(1)
        states = [
            torch.randn(1, STATE_DIM),
            None,
            torch.randn(1, STATE_DIM)
        ]
        actions = self.policy(states)
        tt.assert_equal(actions, torch.tensor([2, 0, 1]))
        self.policy.reinforce(torch.tensor([[1, 2, 3]]).float())

if __name__ == '__main__':
    unittest.main()
