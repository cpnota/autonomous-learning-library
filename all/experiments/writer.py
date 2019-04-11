
import os
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime
from tensorboardX import SummaryWriter


class Writer(ABC):
    @abstractmethod
    def add_loss(self, name, value, step="frame"):
        pass

    @abstractmethod
    def add_evaluation(self, name, value, step="frame"):
        pass

    @abstractmethod
    def add_scalar(self, name, value, step="frame"):
        pass


class DummyWriter(Writer):
    def add_loss(self, name, value, step="frame"):
        pass

    def add_evaluation(self, name, value, step="frame"):
        pass

    def add_scalar(self, name, value, step="frame"):
        pass


class ExperimentWriter(SummaryWriter, Writer):
    def __init__(self, agent_name, env_name):
        self.env_name = env_name
        current_time = str(datetime.now())
        log_dir = os.path.join(
            'runs', ("%s %s %s" % (agent_name, COMMIT_HASH, current_time))
        )
        self._frames = 0
        self._episodes = 1
        super().__init__(log_dir=log_dir)

    def add_loss(self, name, value, step="frame"):
        self.add_scalar("loss/" + name, value, step)

    def add_evaluation(self, name, value, step="frame"):
        self.add_scalar('evaluation/' + name, value, self._get_step(step))

    def add_scalar(self, name, value, step="frgame"):
        super().add_scalar(self.env_name + "/" + name, value, self._get_step(step))

    def _get_step(self, _type):
        if _type == "frame":
            return self.frames
        if _type == "episode":
            return self.episodes
        return _type

    @property
    def frames(self):
        return self._frames

    @frames.setter
    def frames(self, frames):
        self._frames = frames

    @property
    def episodes(self):
        return self._episodes

    @episodes.setter
    def episodes(self, episodes):
        self._episodes = episodes


def get_commit_hash():
    result = subprocess.run(
        ['git', 'rev-parse', '--short', 'HEAD'], stdout=subprocess.PIPE)
    return result.stdout.decode('utf-8').rstrip()


COMMIT_HASH = get_commit_hash()
