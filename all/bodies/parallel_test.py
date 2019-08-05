import unittest
import torch
import torch_testing as tt
from all.agents import Agent
from all.bodies.parallel import ParallelRepeatActions
from all.environments import State

class MockAgent(Agent):
    def __init__(self, n, max_action=6):
        self._actions = torch.zeros(n).long()
        self._max_action = max_action
        self._n = n
        self._states = []
        self._rewards = []

    def act(self, state, reward):
        self._actions = (self._actions + 1) % self._max_action
        self._states.append(state)
        self._rewards.append(reward.view(1, -1))
        return self._actions

# pylint: disable=protected-access


class ParallelRepeatActionsTest(unittest.TestCase):
    def test_repeat_actions(self):
        done = torch.ones(14)
        done[3] = 0
        done[5] = 0
        states = State(
            torch.arange(0, 14),
            done
        )
        rewards = torch.ones(2)

        agent = MockAgent(2)
        body = ParallelRepeatActions(agent, repeats=3)

        actions = body.act(states[0:2], rewards)
        self.assert_array_equal(actions, [1, 1])
        actions = body.act(states[2:4], rewards)
        self.assert_array_equal(actions, [1, None])
        actions = body.act(states[4:6], rewards)
        self.assert_array_equal(actions, [1, None])
        actions = body.act(states[6:8], rewards)
        self.assert_array_equal(actions, [2, 2])
        actions = body.act(states[8:10], rewards)
        self.assert_array_equal(actions, [2, 2])
        actions = body.act(states[10:12], rewards)
        self.assert_array_equal(actions, [2, 2])
        actions = body.act(states[12:14], rewards)
        self.assert_array_equal(actions, [3, 3])

        self.assertEqual(len(agent._states), 3)
        tt.assert_equal(torch.cat(agent._rewards), torch.tensor([
            [1, 1],
            [3, 3],
            [3, 3]
        ]))

    def assert_array_equal(self, actual, expected):
        for i, exp in enumerate(expected):
            self.assertEqual(actual[i], exp, msg=(("\nactual: %s\nexpected: %s")
                                                  % (actual, expected)))

if __name__ == '__main__':
    unittest.main()
