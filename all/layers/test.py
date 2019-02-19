import unittest
import numpy as np
import torch
from torch import nn
from all.layers import Dueling

class TestLayers(unittest.TestCase):
    def test_dueling(self):
        torch.random.manual_seed(0)
        value_model = nn.Linear(2, 1)
        advantage_model = nn.Linear(2, 3)
        model = Dueling(value_model, advantage_model)
        states = torch.tensor([[1., 2.], [3., 4.]])
        result = model(states).detach().numpy()
        np.testing.assert_array_almost_equal(
            result,
            np.array([
                [-0.495295, 0.330573, 0.678836],
                [-1.253222, 1.509323, 2.502186]
            ], dtype=np.float32)
        )

if __name__ == '__main__':
    unittest.main()
