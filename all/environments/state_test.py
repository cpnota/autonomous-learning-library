import unittest
import numpy as np
import torch
import torch_testing as tt
from all.environments.state import State, DONE, NOT_DONE

class StateTest(unittest.TestCase):
    def test_constructor_defaults(self):
        raw = torch.randn(3, 4)
        state = State(raw)
        tt.assert_equal(state.features, raw)
        tt.assert_equal(state.mask, torch.ones(3))
        tt.assert_equal(state.raw, raw)
        self.assertEqual(state.info, [None] * 3)

    def test_custom_constructor_args(self):
        raw = torch.randn(3, 4)
        mask = torch.zeros(3)
        info = ['a', 'b', 'c']
        state = State(raw, mask=mask, info=info)
        tt.assert_equal(state.features, raw)
        tt.assert_equal(state.mask, torch.zeros(3))
        self.assertEqual(state.info, info)

    def test_not_done(self):
        state = State(torch.randn(1, 4))
        self.assertFalse(state.done)

    def test_done(self):
        raw = torch.randn(1, 4)
        state = State(raw, mask=DONE)
        self.assertTrue(state.done)

    def test_from_list(self):
        state1 = State(torch.randn(1, 4), mask=DONE, info=['a'])
        state2 = State(torch.randn(1, 4), mask=NOT_DONE, info=['b'])
        state3 = State(torch.randn(1, 4))
        state = State.from_list([state1, state2, state3])
        tt.assert_equal(state.raw, torch.cat((state1.raw, state2.raw, state3.raw)))
        tt.assert_equal(state.mask, torch.tensor([0, 1, 1]))
        self.assertEqual(state.info, ['a', 'b', None])

    def test_from_gym(self):
        gym_obs = np.array([1, 2, 3])
        done = True
        info = 'a'
        state = State.from_gym(gym_obs, done, info)
        tt.assert_equal(state.raw, torch.tensor([[1, 2, 3]]))
        tt.assert_equal(state.mask, DONE)
        self.assertEqual(state.info, ['a'])

    def test_get_item(self):
        raw = torch.randn(3, 4)
        states = State(raw)
        state = states[2]
        tt.assert_equal(state.raw, raw[2].unsqueeze(0))
        tt.assert_equal(state.mask, NOT_DONE)
        self.assertEqual(state.info, [None])

    def test_len(self):
        state = State(torch.randn(3, 4))
        self.assertEqual(len(state), 3)
