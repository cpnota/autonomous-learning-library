import unittest
import torch
from torch import nn
from torch.nn.functional import smooth_l1_loss
import torch_testing as tt
import numpy as np
from all.environments import State
from all.approximation.q_network import QNetwork

STATE_DIM = 2
ACTIONS = 3

class TestQNetwork(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2)
        self.model = nn.Sequential(
            nn.Linear(STATE_DIM, ACTIONS)
        )
        def optimizer(params):
            return torch.optim.SGD(params, lr=0.1)
        self.q = QNetwork(self.model, optimizer, ACTIONS)

    def test_eval_list(self):
        states = State(
            torch.randn(5, STATE_DIM),
            done=torch.tensor([1, 1, 0, 1, 0])
        )
        result = self.q.eval(states)
        tt.assert_almost_equal(
            result,
            torch.tensor([
                [-0.238509, -0.726287, -0.034026],
                [-0.35688755, -0.6612102, 0.34849477],
                [0., 0., 0.],
                [0.1944, -0.5536, -0.2345],
                [0., 0., 0.]
            ]),
            decimal=2
        )

    def test_eval_actions(self):
        states = State(torch.randn(3, STATE_DIM))
        actions = [1, 2, 0]
        result = self.q.eval(states, actions)
        self.assertEqual(result.shape, torch.Size([3]))
        tt.assert_almost_equal(result, torch.tensor([-0.7262873, 0.3484948, -0.0296164]))


    def test_target_net(self):
        torch.manual_seed(2)
        model = nn.Sequential(
            nn.Linear(1, 1)
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        q = QNetwork(model, optimizer, 1, loss=smooth_l1_loss, target_update_frequency=3)
        inputs = State(torch.tensor([1.]))
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
        np.testing.assert_equal(policy_value, -0.8085841536521912)
        np.testing.assert_equal(target_value, -0.6085841655731201)

if __name__ == '__main__':
    unittest.main()
