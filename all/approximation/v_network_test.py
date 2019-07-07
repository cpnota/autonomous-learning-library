import unittest
import torch
from torch import nn
import torch_testing as tt
from all.approximation.v_network import VNetwork
from all.environments import State

STATE_DIM = 2

class TestVNetwork(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = nn.Sequential(
            nn.Linear(STATE_DIM, 1)
        )

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.v = VNetwork(self.model, optimizer)

    def test_reinforce_list(self):
        states = State(
            torch.randn(5, STATE_DIM),
            mask=torch.tensor([1, 1, 0, 1, 0])
        )
        result = self.v(states)
        tt.assert_almost_equal(result, torch.tensor(
            [0.7053187, 0.3975691, 0., 0.2701665, 0.]))
        self.v.reinforce(torch.tensor([1, -1, 1, 1, 1]).float())
        result = self.v(states)
        tt.assert_almost_equal(result, torch.tensor(
            [0.9732854, 0.5453826, 0., 0.4344811, 0.]))

    def test_multi_reinforce(self):
        states = State(
            torch.randn(5, STATE_DIM),
            mask=torch.tensor([1, 1, 0, 1, 0, 0])
        )
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
