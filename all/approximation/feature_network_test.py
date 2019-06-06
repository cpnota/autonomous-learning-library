import unittest
import torch
from torch import nn
import torch_testing as tt
from all.environments import State
from all.approximation.feature_network import FeatureNetwork

STATE_DIM = 2


class TestFeatureNetwork(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = nn.Sequential(nn.Linear(STATE_DIM, 3))

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.features = FeatureNetwork(self.model, optimizer)
        self.states = State(torch.randn(3, STATE_DIM), mask=torch.tensor([1, 0, 1]))
        self.expected_features = State(
            torch.tensor(
                [
                    [-0.2385, -0.7263, -0.0340],
                    [-0.3569, -0.6612, 0.3485],
                    [-0.0296, -0.7566, -0.4624],
                ]
            ),
            mask=torch.tensor([1, 0, 1]),
        )

    def test_forward(self):
        features = self.features(self.states)
        self.assert_state_equal(features, self.expected_features)

    def test_backward(self):
        states = self.features(self.states)
        loss = torch.tensor(0)
        loss = torch.sum(states.features)
        loss.backward()
        self.features.reinforce()
        features = self.features(self.states)
        expected = State(
            torch.tensor([
                [-0.71, -1.2, -0.5],
                [-0.72, -1.03, -0.02],
                [-0.57, -1.3, -1.01]
            ]),
            mask=torch.tensor([1, 0, 1]),
        )
        self.assert_state_equal(features, expected)

    def test_eval(self):
        features = self.features.eval(self.states)
        self.assert_state_equal(features, self.expected_features)
        self.assertFalse(features.features[0].requires_grad)

    def assert_state_equal(self, actual, expected):
        tt.assert_almost_equal(actual.features, expected.features, decimal=2)
        tt.assert_equal(actual.mask, expected.mask)


if __name__ == "__main__":
    unittest.main()
