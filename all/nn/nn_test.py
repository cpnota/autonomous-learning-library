import unittest
import numpy as np
import torch
import torch_testing as tt
import gym
from all import nn
from all.core import StateArray


class TestNN(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)

    def test_dueling(self):
        torch.random.manual_seed(0)
        value_model = nn.Linear(2, 1)
        advantage_model = nn.Linear(2, 3)
        model = nn.Dueling(value_model, advantage_model)
        states = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        result = model(states).detach().numpy()
        np.testing.assert_array_almost_equal(
            result,
            np.array(
                [[-0.495295, 0.330573, 0.678836], [-1.253222, 1.509323, 2.502186]],
                dtype=np.float32,
            ),
        )

    def test_linear0(self):
        model = nn.Linear0(3, 3)
        result = model(torch.tensor([[3.0, -2.0, 10]]))
        tt.assert_equal(result, torch.tensor([[0.0, 0.0, 0.0]]))

    def test_list(self):
        model = nn.Linear(2, 2)
        net = nn.RLNetwork(model, (2,))
        features = torch.randn((4, 2))
        done = torch.tensor([False, False, True, False])
        out = net(StateArray(features, (4,), done=done))
        tt.assert_almost_equal(
            out,
            torch.tensor(
                [
                    [0.0479387, -0.2268031],
                    [0.2346841, 0.0743403],
                    [0.0, 0.0],
                    [0.2204496, 0.086818],
                ]
            ),
        )

        features = torch.randn(3, 2)
        done = torch.tensor([False, False, False])
        out = net(StateArray(features, (3,), done=done))
        tt.assert_almost_equal(
            out,
            torch.tensor(
                [
                    [0.4234636, 0.1039939],
                    [0.6514298, 0.3354351],
                    [-0.2543002, -0.2041451],
                ]
            ),
        )

    def test_tanh_action_bound(self):
        space = gym.spaces.Box(np.array([-1.0, 10.0]), np.array([1, 20]))
        model = nn.TanhActionBound(space)
        x = torch.tensor([[100.0, 100], [-100, -100], [-100, 100], [0, 0]])
        tt.assert_almost_equal(
            model(x), torch.tensor([[1.0, 20], [-1, 10], [-1, 20], [0.0, 15]])
        )

    def test_categorical_dueling(self):
        n_actions = 2
        n_atoms = 3
        value_model = nn.Linear(2, n_atoms)
        advantage_model = nn.Linear(2, n_actions * n_atoms)
        model = nn.CategoricalDueling(value_model, advantage_model)
        x = torch.randn((2, 2))
        out = model(x)
        self.assertEqual(out.shape, (2, 6))
        tt.assert_almost_equal(
            out,
            torch.tensor(
                [
                    [0.014, -0.691, 0.251, -0.055, -0.419, -0.03],
                    [0.057, -1.172, 0.568, -0.868, -0.482, -0.679],
                ]
            ),
            decimal=3,
        )

    def assert_array_equal(self, actual, expected):
        for first, second in zip(actual, expected):
            if second is None:
                self.assertIsNone(first)
            else:
                tt.assert_almost_equal(first, second, decimal=3)


if __name__ == "__main__":
    unittest.main()
