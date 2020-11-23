import os
import unittest
import torch
from all.environments import GymEnvironment
from all.logging import DummyWriter
from all.presets.classic_control import (
    a2c,
    c51,
    ddqn,
    dqn,
    ppo,
    rainbow,
    vac,
    vpg,
    vqn,
    vsarsa,
)


class TestClassicControlPresets(unittest.TestCase):
    def setUp(self):
        self.env = GymEnvironment('CartPole-v0')
        self.env.reset()

    def tearDown(self):
        if os.path.exists('test_preset.pt'):
            os.remove('test_preset.pt')

    def test_a2c(self):
        self.validate(a2c)

    def test_c51(self):
        self.validate(c51)

    def test_ddqn(self):
        self.validate(ddqn)

    def test_dqn(self):
        self.validate(dqn)

    def test_ppo(self):
        self.validate(ppo)

    def test_rainbow(self):
        self.validate(rainbow)

    def test_vac(self):
        self.validate(vac)

    def test_vpg(self):
        self.validate(vpg)

    def test_vsarsa(self):
        self.validate(vsarsa)

    def test_vqn(self):
        self.validate(vqn)

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
