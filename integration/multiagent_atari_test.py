import unittest
import torch
from all.environments import MultiagentAtariEnv
from all.presets import IndependentMultiagentPreset
from all.presets.atari import dqn
from validate_agent import validate_multiagent


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


class TestMultiagentAtariPresets(unittest.TestCase):
    def test_independent(self):
        env = MultiagentAtariEnv('pong_v2', max_cycles=1000, device=CPU)
        presets = {
            agent_id: dqn.device(CPU).env(env.subenvs[agent_id]).build()
            for agent_id in env.agents
        }
        validate_multiagent(IndependentMultiagentPreset('independent', CPU, presets), env)

    def test_independent_cuda(self):
        env = MultiagentAtariEnv('pong_v2', max_cycles=1000, device=CUDA)
        presets = {
            agent_id: dqn.device(CUDA).env(env.subenvs[agent_id]).build()
            for agent_id in env.agents
        }
        validate_multiagent(IndependentMultiagentPreset('independent', CUDA, presets), env)


if __name__ == "__main__":
    unittest.main()
