import unittest
import numpy as np
from all.optim import Schedulable, LinearScheduler


class Obj(Schedulable):
    def __init__(self):
        self.attr = 0


class TestScheduler(unittest.TestCase):
    def test_linear_scheduler(self):
        obj = Obj()
        obj.attr = LinearScheduler(10, 0, 3, 13)
        expected = [10, 10, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0]
        actual = [obj.attr for _ in expected]
        np.testing.assert_allclose(actual, expected)


if __name__ == '__main__':
    unittest.main()
