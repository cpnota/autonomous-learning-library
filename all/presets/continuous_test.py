import os
import unittest
import torch
from all.core import State
from all.environments import GymEnvironment
from all.logging import DummyWriter
from all.presets.continuous import (
    ddpg,
    ppo,
    sac,
)


class TestContinuousPresets(unittest.TestCase):
    def setUp(self):
        self.env = GymEnvironment('LunarLanderContinuous-v2')
        self.env.reset()

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
        preset = builder().device('cpu').env(self.env).build()
        # normal agent
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


if __name__ == "__main__":
    unittest.main()
