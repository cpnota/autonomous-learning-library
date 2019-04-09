import unittest
import torch
from torch import nn
import torch_testing as tt
from all.approximation.feature_network import FeatureNetwork

STATE_DIM = 2

class TestFeatureNetwork(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = nn.Sequential(
            nn.Linear(STATE_DIM, 3)
        )

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.features = FeatureNetwork(self.model, optimizer)
        self.states = [
            torch.randn(1, STATE_DIM),
            torch.randn(1, STATE_DIM),
            None,
            torch.randn(1, STATE_DIM),
            None
        ]
        self.expected_features = [
            torch.tensor([[-0.2385, -0.7263, -0.0340]]),
            torch.tensor([[-0.3569, -0.6612, 0.3485]]),
            None,
            torch.tensor([[-0.0296, -0.7566, -0.4624]]),
            None
        ]

    def test_forward(self):
        features = self.features(self.states)
        self.assert_array_equal(features, self.expected_features)

    def test_backward(self):
        features = self.features(self.states)
        loss = torch.tensor(0)
        for feature in features:
            if feature is not None:
                loss = torch.sum(feature)
        loss.backward()
        self.features.reinforce()
        features = self.features(self.states)
        self.assert_array_equal(features, [
            torch.tensor([[-0.402, -0.89, -0.197]]),
            torch.tensor([[-0.263, -0.567, 0.442]]),
            None,
            torch.tensor([[-0.505, -1.232, -0.938]]),
            None
        ])

    def test_eval(self):
        features = self.features.eval(self.states)
        self.assert_array_equal(features, self.expected_features)
        self.assertFalse(features[0].requires_grad)

    def assert_array_equal(self, actual, expected):
        for first, second in zip(actual, expected):
            if second is None:
                self.assertIsNone(first)
            else:
                tt.assert_almost_equal(first, second, decimal=3)


if __name__ == '__main__':
    unittest.main()
