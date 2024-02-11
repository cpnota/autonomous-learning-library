import unittest

from pettingzoo.mpe import simple_world_comm_v3

from all.environments import MultiagentPettingZooEnv


class MultiagentPettingZooEnvTest(unittest.TestCase):
    def test_init(self):
        self._make_env()

    def test_reset(self):
        env = self._make_env()
        state = env.reset()
        self.assertEqual(state.observation.shape, (34,))
        self.assertEqual(state.reward, 0)
        self.assertEqual(state.done, False)
        self.assertEqual(state.mask, 1.0)
        self.assertEqual(state["agent"], "leadadversary_0")

    def test_step(self):
        env = self._make_env()
        env.reset()
        state = env.step(0)
        self.assertEqual(state.observation.shape, (34,))
        self.assertEqual(state.reward, 0)
        self.assertEqual(state.done, False)
        self.assertEqual(state.mask, 1.0)
        self.assertEqual(state["agent"], "adversary_0")

    def test_step_tensor(self):
        env = self._make_env()
        env.reset()
        state = env.step(0)
        self.assertEqual(state.observation.shape, (34,))
        self.assertEqual(state.reward, 0)
        self.assertEqual(state.done, False)
        self.assertEqual(state.mask, 1.0)
        self.assertEqual(state["agent"], "adversary_0")

    def test_name(self):
        env = self._make_env()
        self.assertEqual(env.name, "simple_world_comm_v3")

    def test_agent_iter(self):
        env = self._make_env()
        env.reset()
        it = iter(env.agent_iter())
        self.assertEqual(next(it), "leadadversary_0")

    def test_state_spaces(self):
        env = self._make_env()
        self.assertEqual(env.state_space("leadadversary_0").shape, (34,))
        self.assertEqual(env.state_space("adversary_0").shape, (34,))

    def test_action_spaces(self):
        env = self._make_env()
        self.assertEqual(env.action_space("leadadversary_0").n, 20)
        self.assertEqual(env.action_space("adversary_0").n, 5)

    def test_list_agents(self):
        env = self._make_env()
        self.assertEqual(
            env.agents,
            [
                "leadadversary_0",
                "adversary_0",
                "adversary_1",
                "adversary_2",
                "agent_0",
                "agent_1",
            ],
        )

    def test_terminated(self):
        env = self._make_env()
        env.reset()
        self.assertFalse(env.is_done("leadadversary_0"))
        self.assertFalse(env.is_done("adversary_0"))

    def test_last(self):
        env = self._make_env()
        env.reset()
        state = env.last()
        self.assertEqual(state.observation.shape, (34,))
        self.assertEqual(state.reward, 0)
        self.assertEqual(state.done, False)
        self.assertEqual(state.mask, 1.0)
        self.assertEqual(state["agent"], "leadadversary_0")

    def test_variable_spaces(self):
        env = MultiagentPettingZooEnv(
            simple_world_comm_v3.env(), name="simple_world_comm_v2", device="cpu"
        )
        state = env.reset()
        # tests that action spaces work
        for agent in env.agents:
            state = env.last()
            self.assertTrue(
                env.observation_space(agent).contains(
                    state["observation"].cpu().detach().numpy()
                )
            )
            env.step(env.action_space(env.agent_selection).sample())

    def _make_env(self):
        return MultiagentPettingZooEnv(
            simple_world_comm_v3.env(), name="simple_world_comm_v3", device="cpu"
        )


if __name__ == "__main__":
    unittest.main()
