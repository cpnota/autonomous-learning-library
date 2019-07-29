import unittest
import numpy as np
from all.optim import SchedulerMixin, LinearScheduler

class Obj(SchedulerMixin):
    def __init__(self):
        self.a = 0

class TestScheduler(unittest.TestCase):
    def testLinearScheduler(self):
        obj = Obj()
        obj.a = LinearScheduler(10, 0, 3, 13)
        expected = [10, 10, 10, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0]
        actual = [obj.a for _ in expected]
        np.testing.assert_allclose(actual, expected)

if __name__ == '__main__':
    unittest.main()
