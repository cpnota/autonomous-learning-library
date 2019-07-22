import numpy as np
from .runner import SingleEnvRunner, ParallelEnvRunner

class Experiment:
    def __init__(
            self,
            agents,
            envs,
            frames=np.inf,
            episodes=np.inf,
            render=False,
            quiet=False,
            write_loss=True,
    ):
        if not isinstance(agents, list):
            agents = [agents]

        if not isinstance(envs, list):
            envs = [envs]

        for env in envs:
            for agent in agents:
                if isinstance(agent, tuple):
                    runner = ParallelEnvRunner
                else:
                    runner = SingleEnvRunner
                runner(
                    agent,
                    env,
                    frames=frames,
                    episodes=episodes,
                    render=render,
                    quiet=quiet,
                    write_loss=write_loss,
                )
