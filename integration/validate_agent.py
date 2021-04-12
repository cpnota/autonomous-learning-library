import os
from all.logging import DummyWriter
from all.experiments import SingleEnvExperiment, ParallelEnvExperiment, MultiagentEnvExperiment
from all.presets import ParallelPreset, Preset


class TestSingleEnvExperiment(SingleEnvExperiment):
    def _make_writer(self, logdir, agent_name, env_name, write_loss, writer):
        os.makedirs(logdir, exist_ok=True)
        return DummyWriter()


class TestParallelEnvExperiment(ParallelEnvExperiment):
    def _make_writer(self, logdir, agent_name, env_name, write_loss, writer):
        os.makedirs(logdir, exist_ok=True)
        return DummyWriter()


class TestMultiagentEnvExperiment(MultiagentEnvExperiment):
    def _make_writer(self, logdir, agent_name, env_name, write_loss, writer):
        os.makedirs(logdir, exist_ok=True)
        return DummyWriter()


def validate_agent(agent, env):
    preset = agent.env(env).build()
    if isinstance(preset, ParallelPreset):
        experiment = TestParallelEnvExperiment(preset, env, quiet=True)
    else:
        experiment = TestSingleEnvExperiment(preset, env, quiet=True)
    experiment.train(episodes=2)
    experiment.test(episodes=2)


def validate_multiagent(preset, env):
    experiment = TestMultiagentEnvExperiment(preset, env, quiet=True)
    experiment.train(episodes=2)
    experiment.test(episodes=2)
