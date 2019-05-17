import unittest
import torch
import torch_testing as tt
from all.memory import NStepBuffer, NStepBatchBuffer


class NStepBufferTest(unittest.TestCase):
    def test_rollout(self):
        buffer = NStepBuffer(2, discount_factor=0.5)
        actions = torch.ones((3))
        buffer.store(['state1', 'state2', 'state3'], actions, torch.zeros(3))
        buffer.store(['state4', 'state5', 'state6'], actions, torch.ones(3))
        buffer.store(['state7', 'state8', 'state9'], actions, 2 * torch.ones(3))
        buffer.store(['state10', 'state11', 'state12'], actions, 4 * torch.ones(3))
        self.assertEqual(len(buffer), 6)

        states, actions, next_states, returns, lengths = buffer.sample(6)
        expected_states = ['state' + str(i) for i in range(1, 7)]
        expected_next_states = ['state' + str(i) for i in range(7, 13)]
        expected_returns = torch.tensor([
            2, 2, 2,
            4, 4, 4,
        ]).float()
        expected_lengths = torch.tensor([
            2, 2, 2,
            2, 2, 2,
        ])
        self.assert_array_equal(states, expected_states)
        self.assert_array_equal(next_states, expected_next_states)
        tt.assert_allclose(returns, expected_returns)
        tt.assert_equal(lengths, expected_lengths)

    def test_rollout_with_nones(self):
        buffer = NStepBuffer(3, discount_factor=0.5)
        states = [
            ['state1', 'state2', 'state3'],
            ['state4', 'state5', None],
            ['state7', None, 'state9'],
            [None, 'state11', 'state12'],
            ['state13', 'state14', 'state15']
        ]
        actions = torch.ones((3))
        buffer.store(states[0], actions, torch.zeros(3))
        buffer.store(states[1], actions, torch.ones(3))
        buffer.store(states[2], actions, 2 * torch.ones(3))
        buffer.store(states[3], actions, 4 * torch.ones(3))
        buffer.store(states[4], actions, 8 * torch.ones(3))
        states, actions, next_states, returns, lengths = buffer.sample(6)

        expected_states = ['state' + str(i + 1) for i in range(6)]
        expected_states[5] = None
        expect_next_states = [
            None, None, None,
            None, None, None,
        ]
        expected_returns = torch.tensor([
            3, 2, 1,
            4, 2, 0,
        ]).float()
        expected_lengths = torch.tensor([
            3, 2, 1,
            2, 1, 0,
        ])

        self.assert_array_equal(states, expected_states)
        self.assert_array_equal(next_states, expect_next_states)
        tt.assert_allclose(returns, expected_returns)
        tt.assert_equal(lengths, expected_lengths)

    def test_multi_rollout(self):
        buffer = NStepBuffer(2, discount_factor=0.5)
        raw_states = ['state' + str(i) for i in range(12)]
        expected_lengths = torch.tensor([2, 2, 2, 2])
        actions = torch.ones(2)
        buffer.store(raw_states[0:2], actions, torch.ones(2))
        buffer.store(raw_states[2:4], actions, torch.ones(2))
        buffer.store(raw_states[4:6], actions, torch.ones(2))
        buffer.store(raw_states[6:8], actions, torch.ones(2) * 2)

        states, actions, next_states, returns, lengths = buffer.sample(4)
        self.assert_array_equal(states, ['state' + str(i) for i in range(4)])
        self.assert_array_equal(next_states, [
            'state4', 'state5', 'state6', 'state7'
        ])
        tt.assert_allclose(returns, torch.tensor([1.5, 1.5, 2, 2]))
        tt.assert_equal(lengths, expected_lengths)

        buffer.store(raw_states[8:10], actions, torch.ones(2))
        buffer.store(raw_states[10:12], actions, torch.ones(2))

        states, actions, next_states, returns, lengths = buffer.sample(4)
        self.assert_array_equal(
            states, ['state' + str(i) for i in range(4, 8)])
        self.assert_array_equal(next_states, [
            'state8', 'state9', 'state10', 'state11'
        ])
        tt.assert_allclose(returns, torch.tensor([2.5, 2.5, 1.5, 1.5]))
        tt.assert_equal(lengths, expected_lengths)

    def assert_array_equal(self, actual, expected):
        for i, exp in enumerate(expected):
            self.assertEqual(actual[i], exp, msg=(
                ("\nactual: %s\nexpected: %s") % (actual, expected)))

class NStepBatchBufferTest(unittest.TestCase):
    def test_rollout(self):
        buffer = NStepBatchBuffer(2, 3, discount_factor=0.5)
        actions = torch.ones((3))
        buffer.store(['state1', 'state2', 'state3'], actions, torch.zeros(3))
        buffer.store(['state4', 'state5', 'state6'], actions, torch.ones(3))
        buffer.store(['state7', 'state8', 'state9'], actions, 4 * torch.ones(3))
        states, _, next_states, returns, lengths = buffer.sample(-1)

        expected_states = ['state' + str(i + 1) for i in range(6)]
        expect_next_states = [
            'state7', 'state8', 'state9',
            'state7', 'state8', 'state9',
        ]
        expected_returns = torch.tensor([
            3, 3, 3,
            4, 4, 4
        ]).float()
        expected_lengths = torch.tensor([
            2, 2, 2,
            1, 1, 1
        ]).long()

        self.assert_array_equal(states, expected_states)
        self.assert_array_equal(next_states, expect_next_states)
        tt.assert_allclose(returns, expected_returns)
        tt.assert_equal(lengths, expected_lengths)

    def test_rollout_with_nones(self):
        buffer = NStepBatchBuffer(3, 3, discount_factor=0.5)
        states = [
            ['state1', 'state2', 'state3'],
            ['state4', 'state5', None],
            ['state7', None, 'state9'],
            [None, 'state11', 'state12']
        ]
        actions = torch.ones((3))
        buffer.store(states[0], actions, torch.zeros(3))
        buffer.store(states[1], actions, torch.ones(3))
        buffer.store(states[2], actions, 2 * torch.ones(3))
        buffer.store(states[3], actions, 4 * torch.ones(3))
        states, actions, next_states, returns, lengths = buffer.sample(-1)

        expected_states = ['state' + str(i + 1) for i in range(9)]
        expected_states[5] = None
        expected_states[7] = None
        expect_next_states = [
            None, None, None,
            None, None, None,
            None, None, 'state12',
        ]
        expected_returns = torch.tensor([
            3, 2, 1,
            4, 2, 0,
            4, 0, 4
        ]).float()
        expected_lengths = torch.tensor([
            3, 2, 1,
            2, 1, 0,
            1, 0, 1
        ]).float()

        self.assert_array_equal(states, expected_states)
        self.assert_array_equal(next_states, expect_next_states)
        tt.assert_allclose(returns, expected_returns)
        tt.assert_equal(lengths, expected_lengths)

    def test_multi_rollout(self):
        buffer = NStepBatchBuffer(2, 2, discount_factor=0.5)
        raw_states = ['state' + str(i) for i in range(12)]
        actions = torch.ones((2))
        buffer.store(raw_states[0:2], actions, torch.ones(2))
        buffer.store(raw_states[2:4], actions, torch.ones(2))
        buffer.store(raw_states[4:6], actions, torch.ones(2))

        states, actions, next_states, returns, lengths = buffer.sample(-1)
        self.assert_array_equal(states, ['state' + str(i) for i in range(4)])
        self.assert_array_equal(next_states, [
            'state4', 'state5', 'state4', 'state5'
        ])
        tt.assert_allclose(returns, torch.tensor([1.5, 1.5, 1, 1]))
        tt.assert_equal(lengths, torch.tensor([2, 2, 1, 1]))

        buffer.store(raw_states[6:8], actions, torch.ones(2))
        buffer.store(raw_states[8:10], actions, torch.ones(2))

        states, actions, next_states, returns, lengths = buffer.sample(-1)
        self.assert_array_equal(
            states, ['state' + str(i) for i in range(4, 8)])
        self.assert_array_equal(next_states, [
            'state8', 'state9', 'state8', 'state9'
        ])
        tt.assert_allclose(returns, torch.tensor([1.5, 1.5, 1, 1]))
        tt.assert_equal(lengths, torch.tensor([2, 2, 1, 1]))

    def assert_array_equal(self, actual, expected):
        for i, exp in enumerate(expected):
            self.assertEqual(actual[i], exp, msg=(
                ("\nactual: %s\nexpected: %s") % (actual, expected)))


if __name__ == '__main__':
    unittest.main()

if __name__ == '__main__':
    unittest.main()
