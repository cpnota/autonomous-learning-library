
import numpy as np
from .writer import ExperimentWriter
from .experiment import Experiment

class ParallelEnvExperiment(Experiment):
    def __init__(
            self,
            agent,
            env,
            render=False,
            quiet=False,
            write_loss=True
    ):
        super().__init__(ExperimentWriter(self, agent.__name__, env.name, loss=write_loss), quiet)
        self._agent = agent(env, self._writer)
        self._env = env
        self._render = render
        self._frame = 1
        self._episode = 1

    def train(self, frames=np.inf, episodes=np.inf):
        pass

    def test(self, episodes=100):
        pass

    @property
    def frame(self):
        return self._frame

    @property
    def episode(self):
        return self._episode