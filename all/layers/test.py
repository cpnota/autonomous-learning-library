import unittest
import numpy as np
import torch
from torch import nn
import torch_testing as tt
from all.layers import Dueling, Linear0, ListNetwork


class TestLayers(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)

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

    def test_linear0(self):
        model = Linear0(3, 3)
        result = model(torch.tensor([[3., -2., 10]]))
        tt.assert_equal(result, torch.tensor([[0., 0., 0.]]))

    def test_list(self):
        model = nn.Linear(2, 2)
        net = ListNetwork(model, (2))
        x = [torch.randn(1, 2), torch.randn(1, 2), None, torch.randn(1, 2)]
        out = net(x)
        tt.assert_almost_equal(out, torch.tensor([[-2.1408734, -0.553434],
                                                  [-0.499953, -0.0814794],
                                                  [0.,  0.],
                                                  [-0.1632867, 1.5276502]]))


if __name__ == '__main__':
    unittest.main()
