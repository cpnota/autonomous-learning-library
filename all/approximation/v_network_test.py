import unittest
import torch
from torch import nn
import torch_testing as tt
from all.approximation.v_network import ValueNetwork

STATE_DIM = 2

class TestQNetwork(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = nn.Sequential(
            nn.Linear(STATE_DIM, 1)
        )

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.v = ValueNetwork(self.model, optimizer)

    def test_reinforce_list(self):
        states = [
            torch.randn(1, STATE_DIM),
            torch.randn(1, STATE_DIM),
            None,
            torch.randn(1, STATE_DIM),
            None
        ]
        result = self.v(states)
        tt.assert_almost_equal(result, torch.tensor(
            [0.7053187, 0.3975691, 0., 0.0480383, 0.]))
        self.v.reinforce(torch.tensor([1, -1, 1, 1, 1]).float())
        result = self.v(states)
        tt.assert_almost_equal(result, torch.tensor(
            [0.8042525, 0.4606972, 0., 0.0714759, 0.]))

    def test_multi_reinforce(self):
        states = [
            torch.randn(1, STATE_DIM),
            torch.randn(1, STATE_DIM),
            None,
            torch.randn(1, STATE_DIM),
            None,
            None
        ]
        self.v(states[0:2])
        self.v(states[2:4])
        self.v(states[4:6])
        self.v.reinforce(torch.tensor([1, 2]).float())
        self.v.reinforce(torch.tensor([1, 1]).float())
        self.v.reinforce(torch.tensor([1, 2]).float())
        with self.assertRaises(Exception):
            self.v.reinforce(torch.tensor([1, 2]).float())

if __name__ == '__main__':
    unittest.main()
