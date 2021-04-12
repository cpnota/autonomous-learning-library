import unittest
import torch
from torch import nn
import torch_testing as tt
from all.approximation.v_network import VNetwork
from all.core import StateArray

STATE_DIM = 2


def loss(value, error):
    target = value + error
    return ((target.detach() - value) ** 2).mean()


class TestVNetwork(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = nn.Sequential(
            nn.Linear(STATE_DIM, 1)
        )

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.v = VNetwork(self.model, optimizer)

    def test_reinforce_list(self):
        states = StateArray(
            torch.randn(5, STATE_DIM),
            (5,),
            mask=torch.tensor([1, 1, 0, 1, 0])
        )
        result = self.v(states)
        tt.assert_almost_equal(result, torch.tensor([0.7053187, 0.3975691, 0., 0.2701665, 0.]))

        self.v.reinforce(loss(result, torch.tensor([1, -1, 1, 1, 1])).float())
        result = self.v(states)
        tt.assert_almost_equal(result, torch.tensor([0.9732854, 0.5453826, 0., 0.4344811, 0.]))

    def test_multi_reinforce(self):
        states = StateArray(
            torch.randn(6, STATE_DIM),
            (6,),
            mask=torch.tensor([1, 1, 0, 1, 0, 0, 0])
        )
        result1 = self.v(states[0:2])
        self.v.reinforce(loss(result1, torch.tensor([1, 2])).float())
        result2 = self.v(states[2:4])
        self.v.reinforce(loss(result2, torch.tensor([1, 1])).float())
        result3 = self.v(states[4:6])
        self.v.reinforce(loss(result3, torch.tensor([1, 2])).float())
        with self.assertRaises(Exception):
            self.v.reinforce(loss(result3, torch.tensor([1, 2])).float())


if __name__ == '__main__':
    unittest.main()
