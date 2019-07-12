import unittest
import numpy as np
import torch
import torch_testing as tt
import gym
from all import nn
from all.environments import State


class TestNN(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)

    def test_dueling(self):
        torch.random.manual_seed(0)
        value_model = nn.Linear(2, 1)
        advantage_model = nn.Linear(2, 3)
        model = nn.Dueling(value_model, advantage_model)
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
        model = nn.Linear0(3, 3)
        result = model(torch.tensor([[3., -2., 10]]))
        tt.assert_equal(result, torch.tensor([[0., 0., 0.]]))

    def test_list(self):
        model = nn.Linear(2, 2)
        net = nn.ListNetwork(model, (2,))
        features = torch.randn((4, 2))
        done = torch.tensor([1, 1, 0, 1], dtype=torch.uint8)
        out = net(State(features, done))
        tt.assert_almost_equal(out, torch.tensor([[0.0479387, -0.2268031],
                                                  [0.2346841, 0.0743403],
                                                  [0., 0.],
                                                  [0.2204496, 0.086818]]))

        features = torch.randn(3, 2)
        done = torch.tensor([1, 1, 1], dtype=torch.uint8)
        out = net(State(features, done))
        tt.assert_almost_equal(out, torch.tensor([[0.4234636, 0.1039939],
                                                  [0.6514298, 0.3354351],
                                                  [-0.2543002, -0.2041451]]))

    def test_list_to_list(self):
        model = nn.Linear(2, 2)
        net = nn.ListToList(model)
        x = State(torch.randn(5, 2), torch.tensor([1, 1, 1, 0, 1]))
        out = net(x)
        tt.assert_almost_equal(out.features, torch.tensor([
            [0.0479, -0.2268],
            [0.2347, 0.0743],
            [0.0185, 0.0815],
            [0.2204, 0.0868],
            [0.4235, 0.1040]
        ]), decimal=3)
        x = State(torch.randn(3, 2))
        out = net(x)
        tt.assert_almost_equal(out.features, torch.tensor([
            [0.651, 0.335],
            [-0.254, -0.204],
            [0.123, 0.218]
        ]), decimal=3)

        x = State(torch.randn(2))
        out = net(x)
        tt.assert_almost_equal(out.features, torch.tensor([0.3218211, 0.3707529]), decimal=3)

    def test_tanh_action_bound(self):
        space = gym.spaces.Box(
            np.array([-1., 10.]),
            np.array([1, 20])
        )
        model = nn.TanhActionBound(space)
        x = torch.tensor([
            [100., 100],
            [-100, -100],
            [-100, 100],
            [0, 0]
        ])
        tt.assert_almost_equal(model(x), torch.tensor([
            [1., 20],
            [-1, 10],
            [-1, 20],
            [0., 15]
        ]))

    def assert_array_equal(self, actual, expected):
        for first, second in zip(actual, expected):
            if second is None:
                self.assertIsNone(first)
            else:
                tt.assert_almost_equal(first, second, decimal=3)


if __name__ == '__main__':
    unittest.main()
