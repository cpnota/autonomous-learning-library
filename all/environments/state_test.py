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

    def test_not_done(self):
        state = State(torch.randn(1, 4))
        self.assertFalse(state.done)

    def test_done(self):
        raw = torch.randn(1, 4)
        state = State(raw, mask=DONE)
        self.assertTrue(state.done)
