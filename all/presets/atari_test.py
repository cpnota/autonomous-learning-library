import os
import unittest
import torch
from all.environments import AtariEnvironment
from all.logging import DummyWriter
from all.presets.atari import (
    a2c,
    c51,
    ddqn,
    dqn,
    ppo,
    rainbow,
    vac,
    vpg,
    vsarsa,
    vqn
)


class TestAtariPresets(unittest.TestCase):
    def setUp(self):
        self.env = AtariEnvironment('Breakout')
        self.env.reset()

    def tearDown(self):
        if os.path.exists('test_preset.pt'):
            os.remove('test_preset.pt')

    def test_a2c(self):
        self.validate_preset(a2c)

    def test_c51(self):
        self.validate_preset(c51)

    def test_ddqn(self):
        self.validate_preset(ddqn)

    def test_dqn(self):
        self.validate_preset(dqn)

    def test_ppo(self):
        self.validate_preset(ppo)

    def test_rainbow(self):
        self.validate_preset(rainbow)

    def test_vac(self):
        self.validate_preset(vac)

    def test_vpq(self):
        self.validate_preset(vpg)

    def test_vsarsa(self):
        self.validate_preset(vsarsa)

    def test_vqn(self):
        self.validate_preset(vqn)

    def validate_preset(self, builder):
        preset = builder.device('cpu').env(self.env).build()
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
