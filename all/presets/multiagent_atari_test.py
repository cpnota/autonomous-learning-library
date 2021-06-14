import os
import unittest
import torch
from all.environments import MultiagentAtariEnv
from all.logging import DummyWriter
from all.presets.atari import dqn
from all.presets import IndependentMultiagentPreset


class TestMultiagentAtariPresets(unittest.TestCase):
    def setUp(self):
        self.env = MultiagentAtariEnv('pong_v2', device='cpu')
        self.env.reset()

    def tearDown(self):
        if os.path.exists('test_preset.pt'):
            os.remove('test_preset.pt')

    def test_independent(self):
        env = MultiagentAtariEnv('pong_v2', device='cpu')
        presets = {
            agent_id: dqn.device('cpu').env(env.subenvs[agent_id]).build()
            for agent_id in env.agents
        }
        self.validate_preset(IndependentMultiagentPreset('independent', 'cpu', presets), env)

    def validate_preset(self, preset, env):
        # normal agent
        agent = preset.agent(writer=DummyWriter(), train_steps=100000)
        agent.act(self.env.last())
        # test agent
        test_agent = preset.test_agent()
        test_agent.act(self.env.last())
        # test save/load
        preset.save('test_preset.pt')
        preset = torch.load('test_preset.pt')
        test_agent = preset.test_agent()
        test_agent.act(self.env.last())


if __name__ == "__main__":
    unittest.main()
