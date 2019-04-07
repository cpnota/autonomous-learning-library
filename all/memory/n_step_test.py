import unittest
import torch
import torch_testing as tt
from all.memory import NStepBuffer

class NStepBufferTest(unittest.TestCase):
    def test_rollout(self):
        buffer = NStepBuffer(3, discount=0.5)
        buffer.store(['state1', 'state2', 'state3'], torch.zeros(3))
        buffer.store(['state4', 'state5', 'state6'], torch.ones(3))
        buffer.store(['state7', 'state8', 'state9'], 2 * torch.ones(3))
        buffer.store(['state10', 'state11', 'state12'], 4 * torch.ones(3))
        states, next_states, returns = buffer.sample(-1)

        expected_states = ['state' + str(i + 1) for i in range(9)]
        expect_next_states = [
            'state10', 'state11', 'state12',
            'state10', 'state11', 'state12',
            'state10', 'state11', 'state12',
        ]
        expected_returns = torch.tensor([
            3, 3, 3,
            4, 4, 4,
            4, 4, 4
        ]).float()

        self.assert_array_equal(states, expected_states)
        self.assert_array_equal(next_states, expect_next_states)
        tt.assert_allclose(returns, expected_returns)

    def assert_array_equal(self, actual, expected):
        for i, exp in enumerate(expected):
            self.assertEqual(actual[i], exp)

    def test_rollout_with_nones(self):
        buffer = NStepBuffer(3, discount=0.5)
        states = [
            ['state1', 'state2', 'state3'],
            ['state4', 'state5', None],
            ['state7', None, 'state9'],
            [None, 'state11', 'state12']
        ]
        buffer.store(states[0], torch.zeros(3))
        buffer.store(states[1], torch.ones(3))
        buffer.store(states[2], 2 * torch.ones(3))
        buffer.store(states[3], 4 * torch.ones(3))
        states, next_states, returns = buffer.sample(-1)

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

        self.assert_array_equal(states, expected_states)
        self.assert_array_equal(next_states, expect_next_states)
        tt.assert_allclose(returns, expected_returns)

if __name__ == '__main__':
    unittest.main()
