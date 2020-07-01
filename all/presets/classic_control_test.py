import unittest
from all.environments import GymEnvironment
from all.presets.validate_agent import validate_agent
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
    def test_a2c(self):
        self.validate(a2c())

    def test_c51(self):
        self.validate(c51())

    def test_ddqn(self):
        self.validate(ddqn())

    def test_dqn(self):
        self.validate(dqn())

    def test_ppo(self):
        self.validate(ppo())

    def test_rainbow(self):
        self.validate(rainbow())

    def test_vac(self):
        self.validate(vac())

    def test_vpg(self):
        self.validate(vpg())

    def test_vsarsa(self):
        self.validate(vsarsa())

    def test_vqn(self):
        self.validate(vqn())

    def validate(self, make_agent):
        validate_agent(make_agent, GymEnvironment("CartPole-v0"))


if __name__ == "__main__":
    unittest.main()
