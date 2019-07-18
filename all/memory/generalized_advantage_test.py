import unittest
import torch
import torch_testing as tt
from all import nn
from all.environments import State
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
        states = State(torch.arange(0, 3).unsqueeze(1))
        rewards = torch.tensor([1., 2, 4])
        buffer.store(states[0], actions, rewards[0])
        buffer.store(states[1], actions, rewards[1])
        buffer.store(states[2], actions, rewards[2])

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

        _states, _actions, _advantages = buffer.sample(-1)
        tt.assert_almost_equal(_advantages, advantages)
        tt.assert_equal(_actions, torch.tensor([1, 1]))

    def test_multi(self):
        buffer = GeneralizedAdvantageBuffer(
            self.v,
            self.features,
            2,
            2,
            discount_factor=0.5,
            lam=0.5
        )
        actions = torch.ones((2))
        states = [
            State(torch.tensor([[0], [3]])),
            State(torch.tensor([[1], [4]])),
            State(torch.tensor([[2], [5]])),
        ]
        rewards = torch.tensor([[1., 1], [2, 1], [4, 1]])
        buffer.store(states[0], actions, rewards[0])
        buffer.store(states[1], actions, rewards[1])
        buffer.store(states[2], actions, rewards[2])

        states = State.from_list(states)
        values = self.v.eval(self.features.eval(states)).view(3, -1)
        tt.assert_almost_equal(values, torch.tensor([
            [ 0.183, -1.408],
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

        _states, _actions, _advantages = buffer.sample(-1)
        tt.assert_almost_equal(_advantages, advantages.view(-1))

    # def test_rollout(self):
    #     buffer = GeneralizedAdvantageBuffer(self.v, self.features, 2, 3, discount_factor=0.5)
    #     actions = torch.ones((3))
    #     states = State(torch.arange(0, 12).unsqueeze(1))
    #     buffer.store(states[0:3], actions, torch.zeros(3))
    #     buffer.store(states[3:6], actions, torch.ones(3))
    #     buffer.store(states[6:9], actions, 4 * torch.ones(3))
    #     states, _, advantages = buffer.sample(-1)

    #     expected_states = State(torch.arange(0, 6).unsqueeze(1))
    #     expected_next_states = State(
    #         torch.cat((torch.arange(6, 9), torch.arange(6, 9))).unsqueeze(1)
    #     )
    #     expected_returns = torch.tensor([
    #         0.5, 0.5, 0.5,
    #         1, 1, 1
    #     ]).float()
    #     expected_lengths = torch.tensor([
    #         2., 2, 2,
    #         1, 1, 1
    #     ])


    #     self.assert_states_equal(states, expected_states)
    #     tt.assert_allclose(advantages, self._compute_expected_advantages(
    #         expected_states, expected_returns, expected_next_states, expected_lengths
    #     ))

    # def test_rollout_with_nones(self):
    #     buffer = GeneralizedAdvantageBuffer(self.v, self.features, 3, 3, discount_factor=0.5)
    #     done = torch.ones(12)
    #     done[5] = 0
    #     done[7] = 0
    #     done[9] = 0
    #     states = State(torch.arange(0, 12).unsqueeze(1), done)
    #     actions = torch.ones((3))
    #     buffer.store(states[0:3], actions, torch.zeros(3))
    #     buffer.store(states[3:6], actions, torch.ones(3))
    #     buffer.store(states[6:9], actions, 2 * torch.ones(3))
    #     buffer.store(states[9:12], actions, 4 * torch.ones(3))
    #     states, actions, advantages = buffer.sample(-1)

    #     expected_states = State(torch.arange(0, 9).unsqueeze(1), done[0:9])
    #     expected_next_done = torch.zeros(9)
    #     expected_next_done[5] = 1
    #     expected_next_done[7] = 1
    #     expected_next_done[8] = 1
    #     expected_next_states = State(torch.tensor([
    #         9, 7, 5,
    #         9, 7, 11,
    #         9, 10, 11
    #     ]).unsqueeze(1), expected_next_done)
    #     expected_returns = torch.tensor([
    #         1, 0.5, 0,
    #         2, 1, 2,
    #         2, 2, 2
    #     ]).float()
    #     expected_lengths = torch.tensor([
    #         3, 2, 1,
    #         2, 1, 2,
    #         1, 1, 1
    #     ]).float()

    #     self.assert_states_equal(states, expected_states)
    #     tt.assert_allclose(advantages, self._compute_expected_advantages(
    #         expected_states, expected_returns, expected_next_states, expected_lengths
    #     ))

    # def test_multi_rollout(self):
    #     buffer = GeneralizedAdvantageBuffer(self.v, self.features, 2, 2, discount_factor=0.5)
    #     raw_states = State(torch.arange(0, 12).unsqueeze(1))
    #     actions = torch.ones((2))
    #     buffer.store(raw_states[0:2], actions, torch.ones(2))
    #     buffer.store(raw_states[2:4], actions, torch.ones(2))
    #     buffer.store(raw_states[4:6], actions, torch.ones(2))

    #     states, actions, advantages = buffer.sample(-1)
    #     expected_states = State(torch.arange(0, 4).unsqueeze(1))
    #     expected_returns = torch.tensor([1.5, 1.5, 1, 1])
    #     expected_next_states = State(torch.tensor([4, 5, 4, 5]).unsqueeze(1))
    #     expected_lengths = torch.tensor([2., 2, 1, 1])
    #     self.assert_states_equal(states, expected_states)
    #     tt.assert_allclose(advantages, self._compute_expected_advantages(
    #         expected_states,
    #         expected_returns,
    #         expected_next_states,
    #         expected_lengths
    #     ))

    #     buffer.store(raw_states[6:8], actions, torch.ones(2))
    #     buffer.store(raw_states[8:10], actions, torch.ones(2))

    #     states, actions, advantages = buffer.sample(-1)
    #     expected_states = State(torch.arange(4, 8).unsqueeze(1))
    #     self.assert_states_equal(states, expected_states)
    #     tt.assert_allclose(advantages, self._compute_expected_advantages(
    #         expected_states,
    #         torch.tensor([1.5, 1.5, 1, 1]),
    #         State(torch.tensor([8, 9, 8, 9]).unsqueeze(1)),
    #         torch.tensor([2., 2, 1, 1])
    #     ))

    def assert_array_equal(self, actual, expected):
        for i, exp in enumerate(expected):
            self.assertEqual(actual[i], exp, msg=(
                ("\nactual: %s\nexpected: %s") % (actual, expected)))

    def assert_states_equal(self, actual, expected):
        tt.assert_almost_equal(actual.raw, expected.raw)
        tt.assert_equal(actual.mask, expected.mask)


if __name__ == '__main__':
    unittest.main()
