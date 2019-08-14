import unittest
import torch
from torch import nn
from torch.nn.functional import smooth_l1_loss
import torch_testing as tt
import numpy as np
from all.environments import State
from all.approximation import QDist, FixedTarget

STATE_DIM = 1
ACTIONS = 2
ATOMS = 5
V_MIN = -2
V_MAX = 2

class TestQDist(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = nn.Sequential(
            nn.Linear(STATE_DIM, ACTIONS * ATOMS)
        )
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.q = QDist(
            self.model,
            optimizer,
            ACTIONS,
            ATOMS,
            V_MIN,
            V_MAX
        )

    def test_atoms(self):
        tt.assert_almost_equal(self.q.atoms, torch.tensor([-2, -1, 0, 1, 2]))

if __name__ == '__main__':
    unittest.main()
