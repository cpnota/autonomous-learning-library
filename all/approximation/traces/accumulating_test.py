import unittest
from gym.spaces import Box
import numpy as np
from all.approximation.value.action import LinearStateDiscreteActionValue
from all.approximation.bases.fourier import FourierBasis
from all.approximation.traces import AccumulatingTraces


SPACE = Box(low=0, high=1, shape=(2,), dtype=np.float32)
STATE = np.array([0.5, 1])


class Env:
    def __init__(self):
        self.done = False

    def new_episode(self):
        pass

    def step(self):
        pass

class TestAccumulatingTraces(unittest.TestCase):
    def setUp(self):
        self.basis = FourierBasis(SPACE, 2)
        self.approximation = LinearStateDiscreteActionValue(
            0.1, self.basis, actions=3)
        self.env = Env()
        self.traces = AccumulatingTraces(self.approximation, self.env, 0.5)

    def test_init(self):
        np.testing.assert_equal(self.traces.call(STATE), np.array([0, 0, 0]))

    def test_update_once(self):
        np.testing.assert_equal(self.traces.call(STATE), np.array([0, 0, 0]))
        self.traces.update(1, STATE, 1)
        np.testing.assert_allclose(
            self.traces.call(STATE), np.array([0, 0.6, 0]))

    def test_call_one(self):
        np.testing.assert_equal(self.traces.call(STATE), np.array([0, 0, 0]))
        self.traces.update(1, STATE, 1)
        np.testing.assert_approx_equal(self.traces.call(STATE, 1), 0.6)

    def test_update_multiple(self):
        np.testing.assert_equal(self.traces.call(STATE), np.array([0, 0, 0]))
        self.traces.update(1, STATE, 1)
        self.traces.update(1, STATE, 1)
        # should be 0.6 + (0.6 + 0.3)
        np.testing.assert_allclose(
            self.traces.call(STATE), np.array([0, 1.5, 0]))
        self.traces.update(1, STATE, 2)
        np.testing.assert_allclose(
            self.traces.call(STATE), np.array([0, 1.95, 0.6]))

    def test_clears_traces_on_done(self):
        np.testing.assert_equal(self.traces.call(STATE), np.array([0, 0, 0]))
        self.traces.update(1, STATE, 1)
        self.env.done = True
        self.traces.update(1, STATE, 1)
        # should be 0.6 + (0.6 + 0.3)
        np.testing.assert_allclose(
            self.traces.call(STATE), np.array([0, 1.5, 0]))
        self.traces.update(1, STATE, 2)
        # these should be cleared compared to above
        np.testing.assert_allclose(
            self.traces.call(STATE), np.array([0, 1.5, 0.6]))


if __name__ == '__main__':
    unittest.main()
