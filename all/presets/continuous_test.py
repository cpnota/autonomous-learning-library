import os
import unittest
import torch
from all.core import State
from all.environments import GymEnvironment, DuplicateEnvironment
from all.logging import DummyWriter
from all.presets import Preset, ParallelPreset
from all.presets.continuous import (
    ddpg,
    ppo,
    sac,
)


class TestContinuousPresets(unittest.TestCase):
    def setUp(self):
        self.env = GymEnvironment('LunarLanderContinuous-v2')
        self.env.reset()
        self.parallel_env = DuplicateEnvironment([
            GymEnvironment('LunarLanderContinuous-v2'),
            GymEnvironment('LunarLanderContinuous-v2'),
        ])
        self.parallel_env.reset()

    def tearDown(self):
        if os.path.exists('test_preset.pt'):
            os.remove('test_preset.pt')

    def test_ddpg(self):
        self.validate(ddpg)

    def test_ppo(self):
        self.validate(ppo)

    def test_sac(self):
        self.validate(sac)

    def validate(self, builder):
        preset = builder.device('cpu').env(self.env).build()
        if isinstance(preset, ParallelPreset):
            return self.validate_parallel_preset(preset)
        return self.validate_standard_preset(preset)

    def validate_standard_preset(self, preset):
        # train agent
        agent = preset.agent(writer=DummyWriter(), train_steps=100000)
        agent.act(self.env.state)
        # test agent
        test_agent = preset.test_agent()
        test_agent.act(self.env.state)
        # test save/load
        preset.save('test_preset.pt')
        preset = torch.load('test_preset.pt')
        test_agent = preset.test_agent()
        test_agent.act(self.env.state)

    def validate_parallel_preset(self, preset):
        # train agent
        agent = preset.agent(writer=DummyWriter(), train_steps=100000)
        agent.act(self.parallel_env.state_array)
        # test agent
        test_agent = preset.test_agent()
        test_agent.act(self.env.state)
        # parallel test_agent
        parallel_test_agent = preset.test_agent()
        parallel_test_agent.act(self.parallel_env.state_array)
        # test save/load
        preset.save('test_preset.pt')
        preset = torch.load('test_preset.pt')
        test_agent = preset.test_agent()
        test_agent.act(self.env.state)


if __name__ == "__main__":
    unittest.main()
