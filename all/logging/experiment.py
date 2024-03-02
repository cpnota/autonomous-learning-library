import csv
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from ._logger import Logger


class ExperimentLogger(SummaryWriter, Logger):
    """
    The default Logger object used by all.experiments.Experiment.
    Writes logs using tensorboard into the current logdir directory ('runs' by default),
    tagging the run with a combination of the agent name, the commit hash of the
    current git repo of the working directory (if any), and the current time.
    Also writes summary statistics into CSV files.
    Args:
        experiment (all.experiments.Experiment): The Experiment associated with the Logger object.
        agent_name (str): The name of the Agent the Experiment is being performed on
        env_name (str): The name of the environment the Experiment is being performed in
        verbose (bool, optional): Whether or not to log all data or only summary metrics.
    """

    def __init__(self, experiment, agent_name, env_name, verbose=True, logdir="runs"):
        self.env_name = env_name
        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S_%f")
        dir_name = f"{agent_name}_{env_name}_{current_time}"
        os.makedirs(os.path.join(logdir, dir_name))
        self.log_dir = os.path.join(logdir, dir_name)
        self._experiment = experiment
        self._verbose = verbose
        super().__init__(log_dir=self.log_dir)

    def add_summary(self, name, values, step="frame"):
        aggregators = ["mean", "std", "max", "min"]
        metrics = {
            aggregator: getattr(np, aggregator)(values) for aggregator in aggregators
        }
        for aggregator, value in metrics.items():
            super().add_scalar(
                f"summary/{name}/{aggregator}", value, self._get_step(step)
            )

        # log summary statistics to file
        with open(os.path.join(self.log_dir, name + ".csv"), "a") as csvfile:
            csv.writer(csvfile).writerow([self._get_step(step), *metrics.values()])

    def add_loss(self, name, value, step="frame"):
        self._add_scalar("loss/" + name, value, step)

    def add_eval(self, name, value, step="frame"):
        self._add_scalar("eval/" + name, value, step)

    def add_info(self, name, value, step="frame"):
        self._add_scalar("info/" + name, value, step)

    def add_schedule(self, name, value, step="frame"):
        self._add_scalar("schedule/" + name, value, step)

    def add_hparams(self, hparam_dict, metric_dict, step="frame"):
        allowed_types = (int, float, str, bool, torch.Tensor)
        hparams = {k: v for k, v in hparam_dict.items() if isinstance(v, allowed_types)}
        super().add_hparams(
            hparams, metric_dict, run_name=".", global_step=self._get_step("frame")
        )

    def _add_scalar(self, name, value, step="frame"):
        if self._verbose:
            super().add_scalar(name, value, self._get_step(step))

    def _get_step(self, _type):
        if _type == "frame":
            return self._experiment.frame
        if _type == "episode":
            return self._experiment.episode
        return _type

    def close(self):
        pass
