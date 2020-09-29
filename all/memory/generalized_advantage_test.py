import unittest
import torch
import torch_testing as tt
from all import nn
from all.core import State
from all.approximation import VNetwork, FeatureNetwork
from all.memory import GeneralizedAdvantageBuffer


class GeneralizedAdvantageBufferTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.features = FeatureNetwork(nn.Linear(1, 2), None)
        self.v = VNetwork(nn.Linear(2, 1), None)

    def _compute_expected_advantages(self, states, returns, next_states, lengths):
        return (
            returns
            + (0.5 ** lengths) * self.v.eval(self.features.eval(next_states))
            - self.v.eval(self.features.eval(states))
        )

    def test_simple(self):
        buffer = GeneralizedAdvantageBuffer(
            self.v,
            self.features,
            2,
            1,
            discount_factor=0.5,
            lam=0.5
        )
        actions = torch.ones((1))
        states = State.array([State({'observation': torch.tensor([float(x)])}) for x in range(3)])
        rewards = torch.tensor([1., 2, 4])
        buffer.store(states[0], actions, rewards[0])
        buffer.store(states[1], actions, rewards[1])

        values = self.v.eval(self.features.eval(states))
        tt.assert_almost_equal(values, torch.tensor([0.1826, -0.3476, -0.8777]), decimal=3)

        td_errors = torch.zeros(2)
        td_errors[0] = rewards[0] + 0.5 * values[1] - values[0]
        td_errors[1] = rewards[1] + 0.5 * values[2] - values[1]
        tt.assert_almost_equal(td_errors, torch.tensor([0.6436, 1.909]), decimal=3)

        advantages = torch.zeros(2)
        advantages[0] = td_errors[0] + 0.25 * td_errors[1]
        advantages[1] = td_errors[1]
        tt.assert_almost_equal(advantages, torch.tensor([1.121, 1.909]), decimal=3)

        _states, _actions, _advantages = buffer.advantages(states[2])
        tt.assert_almost_equal(_advantages, advantages)
        tt.assert_equal(_actions, torch.tensor([1, 1]))

    def test_parallel(self):
        buffer = GeneralizedAdvantageBuffer(
            self.v,
            self.features,
            2,
            2,
            discount_factor=0.5,
            lam=0.5
        )
        actions = torch.ones((2))

        def make_states(x, y):
            return State.array([
                State({'observation': torch.tensor([float(x)])}),
                State({'observation': torch.tensor([float(y)])})
            ])

        states = State.array([
            make_states(0, 3),
            make_states(1, 4),
            make_states(2, 5),
        ])
        self.assertEqual(states.shape, (3, 2))
        rewards = torch.tensor([[1., 1], [2, 1], [4, 1]])
        buffer.store(states[0], actions, rewards[0])
        buffer.store(states[1], actions, rewards[1])

        values = self.v.eval(self.features.eval(states)).view(3, -1)
        tt.assert_almost_equal(values, torch.tensor([
            [0.183, -1.408],
            [-0.348, -1.938],
            [-0.878, -2.468]
        ]), decimal=3)

        td_errors = torch.zeros(2, 2)
        td_errors[0] = rewards[0] + 0.5 * values[1] - values[0]
        td_errors[1] = rewards[1] + 0.5 * values[2] - values[1]
        tt.assert_almost_equal(td_errors, torch.tensor([
            [0.6436, 1.439],
            [1.909, 1.704]
        ]), decimal=3)

        advantages = torch.zeros(2, 2)
        advantages[0] = td_errors[0] + 0.25 * td_errors[1]
        advantages[1] = td_errors[1]
        tt.assert_almost_equal(advantages, torch.tensor([
            [1.121, 1.865],
            [1.909, 1.704]
        ]), decimal=3)

        _states, _actions, _advantages = buffer.advantages(states[2])
        tt.assert_almost_equal(_advantages, advantages.view(-1))

    def assert_array_equal(self, actual, expected):
        for i, exp in enumerate(expected):
            self.assertEqual(actual[i], exp, msg=(
                ("\nactual: %s\nexpected: %s") % (actual, expected)))

    def assert_states_equal(self, actual, expected):
        tt.assert_almost_equal(actual.raw, expected.raw)
        tt.assert_equal(actual.mask, expected.mask)


if __name__ == '__main__':
    unittest.main()
