import unittest
import torch
from torch import nn
import numpy as np
from all.approximation.q_tabular import QTabular

STATE_DIM = 2
ACTIONS = 3


class TestTabular(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = nn.Sequential(
            nn.Linear(STATE_DIM, ACTIONS)
        )
        def optimizer(params):
            return torch.optim.SGD(params, lr=0.1)
        self.q = QTabular(self.model, optimizer)

    def test_eval_list(self):
        states = [
            torch.randn(1, STATE_DIM),
            torch.randn(1, STATE_DIM),
            None,
            torch.randn(1, STATE_DIM),
            None
        ]
        result = self.q.eval(states)
        np.testing.assert_array_almost_equal(
            result.detach().numpy(),
            np.array([[-0.238509, -0.726287, -0.034026],
                      [-0.35688755, -0.6612102, 0.34849477],
                      [0., 0., 0.],
                      [-0.02961645, -0.7566322, -0.46243042],
                      [0., 0., 0.]], dtype=np.float32)
        )

    def test_target_net(self):
        torch.manual_seed(2)
        model = nn.Sequential(
            nn.Linear(1, 1)
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        q = QTabular(model, optimizer, target_update_frequency=3)
        inputs = torch.tensor([1.])
        errors = torch.tensor([-1.])

        policy_value = q(inputs).item()
        target_value = q.eval(inputs).item()
        np.testing.assert_equal(policy_value, -0.008584141731262207)
        np.testing.assert_equal(target_value, -0.008584141731262207)

        q.reinforce(errors)
        policy_value = q(inputs).item()
        target_value = q.eval(inputs).item()
        np.testing.assert_equal(policy_value, -0.20858412981033325)
        np.testing.assert_equal(target_value, -0.008584141731262207)

        q.reinforce(errors)
        policy_value = q(inputs).item()
        target_value = q.eval(inputs).item()
        np.testing.assert_equal(policy_value, -0.4085841178894043)
        np.testing.assert_equal(target_value, -0.008584141731262207)

        q.reinforce(errors)
        policy_value = q(inputs).item()
        target_value = q.eval(inputs).item()
        np.testing.assert_equal(policy_value, -0.6085841655731201)
        np.testing.assert_equal(target_value, -0.6085841655731201)

        q.reinforce(errors)
        policy_value = q(inputs).item()
        target_value = q.eval(inputs).item()
        np.testing.assert_equal(policy_value, -0.8085842132568359)
        np.testing.assert_equal(target_value, -0.6085841655731201)

if __name__ == '__main__':
    unittest.main()
