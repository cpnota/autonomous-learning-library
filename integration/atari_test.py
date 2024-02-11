import unittest

import torch
from validate_agent import validate_agent

from all.environments import AtariEnvironment
from all.presets.atari import a2c, c51, ddqn, dqn, ppo, rainbow, vac, vpg, vqn, vsarsa

CPU = torch.device("cpu")
if torch.cuda.is_available():
    CUDA = torch.device("cuda")
else:
    print(
        "WARNING: CUDA is not available!",
        "Running presets in cpu mode.",
        "Enable CUDA for full test coverage!",
    )
    CUDA = torch.device("cpu")


class TestAtariPresets(unittest.TestCase):
    def test_a2c(self):
        validate_agent(a2c.device(CPU), AtariEnvironment("Breakout", device=CPU))

    def test_a2c_cuda(self):
        validate_agent(a2c.device(CUDA), AtariEnvironment("Breakout", device=CUDA))

    def test_c51(self):
        validate_agent(c51.device(CPU), AtariEnvironment("Breakout", device=CPU))

    def test_c51_cuda(self):
        validate_agent(c51.device(CUDA), AtariEnvironment("Breakout", device=CUDA))

    def test_ddqn(self):
        validate_agent(
            ddqn.device(CPU),
            AtariEnvironment("Breakout", device=CPU),
        )

    def test_ddqn_cuda(self):
        validate_agent(
            ddqn.device(CUDA).hyperparameters(replay_start_size=64),
            AtariEnvironment("Breakout", device=CUDA),
        )

    def test_dqn(self):
        validate_agent(
            dqn.device(CPU).hyperparameters(replay_start_size=64),
            AtariEnvironment("Breakout", device=CPU),
        )

    def test_dqn_cuda(self):
        validate_agent(
            dqn.device(CUDA).hyperparameters(replay_start_size=64),
            AtariEnvironment("Breakout", device=CUDA),
        )

    def test_ppo(self):
        validate_agent(
            ppo.device(CPU).hyperparameters(n_envs=4),
            AtariEnvironment("Breakout", device=CPU),
        )

    def test_ppo_cuda(self):
        validate_agent(
            ppo.device(CUDA).hyperparameters(n_envs=4),
            AtariEnvironment("Breakout", device=CUDA),
        )

    def test_rainbow(self):
        validate_agent(
            rainbow.device(CPU).hyperparameters(replay_start_size=64),
            AtariEnvironment("Breakout", device=CPU),
        )

    def test_rainbow_cuda(self):
        validate_agent(
            rainbow.device(CUDA).hyperparameters(replay_start_size=64),
            AtariEnvironment("Breakout", device=CUDA),
        )

    def test_vac(self):
        validate_agent(
            vac.device(CPU).hyperparameters(n_envs=4),
            AtariEnvironment("Breakout", device=CPU),
        )

    def test_vac_cuda(self):
        validate_agent(
            vac.device(CUDA).hyperparameters(n_envs=4),
            AtariEnvironment("Breakout", device=CUDA),
        )

    def test_vpg(self):
        validate_agent(vpg.device(CPU), AtariEnvironment("Breakout", device=CPU))

    def test_vpg_cuda(self):
        validate_agent(vpg.device(CUDA), AtariEnvironment("Breakout", device=CUDA))

    def test_vsarsa(self):
        validate_agent(
            vsarsa.device(CPU).hyperparameters(n_envs=4),
            AtariEnvironment("Breakout", device=CPU),
        )

    def test_vsarsa_cuda(self):
        validate_agent(
            vsarsa.device(CUDA).hyperparameters(n_envs=4),
            AtariEnvironment("Breakout", device=CUDA),
        )

    def test_vqn(self):
        validate_agent(
            vqn.device(CPU).hyperparameters(n_envs=4),
            AtariEnvironment("Breakout", device=CPU),
        )

    def test_vqn_cuda(self):
        validate_agent(
            vqn.device(CUDA).hyperparameters(n_envs=4),
            AtariEnvironment("Breakout", device=CUDA),
        )


if __name__ == "__main__":
    unittest.main()
