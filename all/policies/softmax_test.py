import unittest
import os
import numpy as np
import torch
from torch import nn
from . import SoftmaxPolicy

STATE_DIM = 2
ACTIONS = 3

class TestSoftmax(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = nn.Sequential(
            nn.Linear(STATE_DIM, ACTIONS)
        )
        def optimizer(params):
            return torch.optim.SGD(params, lr=0.1)
        self.policy = SoftmaxPolicy(self.model, optimizer)

    def test_run(self):
        state = torch.randn(STATE_DIM)
        action = self.policy.execute(state)
        self.assertEqual(action.item(), 0)
        state = torch.randn(STATE_DIM)
        action = self.policy.execute(state)
        self.assertEqual(action.item(), 2)
        self.policy.reinforce(torch.tensor([-1, 1000000]).float())
        action = self.policy.execute(state)
        self.assertEqual(action.item(), 2)

if __name__ == '__main__':
    unittest.main()
