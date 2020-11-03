
import os
import csv
import subprocess
from datetime import datetime
from tensorboardX import SummaryWriter
from all.logging import Writer


class ExperimentWriter(SummaryWriter, Writer):
    '''
    The Writer object used by all.experiments.Experiment.
    Writes logs using tensorboard into the current logdir directory ('runs' by default),
    tagging the run with a combination of the agent name, the commit hash of the
    current git repo of the working directory (if any), and the current time.
    Also writes summary statistics into CSV files.
    Args:
        experiment (all.experiments.Experiment): The Experiment associated with the Writer object.
        agent_name (str): The name of the Agent the Experiment is being performed on
        env_name (str): The name of the environment the Experiment is being performed in
        loss (bool, optional): Whether or not to log loss/scheduling metrics, or only evaluation and summary metrics.
    '''
    def __init__(self, experiment, agent_name, env_name, loss=True, logdir='runs'):
        self.env_name = env_name
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S %f')
        os.makedirs(
            os.path.join(
                logdir, ("%s %s %s" % (agent_name, COMMIT_HASH, current_time)), env_name
            )
        )
        self.log_dir = os.path.join(
            logdir, ("%s %s %s" % (agent_name, COMMIT_HASH, current_time))
        )
        self._experiment = experiment
        self._loss = loss
        super().__init__(log_dir=self.log_dir)

    def add_loss(self, name, value, step="frame"):
        if self._loss:
            self.add_scalar("loss/" + name, value, step)

    def add_evaluation(self, name, value, step="frame"):
        self.add_scalar("evaluation/" + name, value, self._get_step(step))

    def add_schedule(self, name, value, step="frame"):
        if self._loss:
            self.add_scalar("schedule" + "/" + name, value, self._get_step(step))

    def add_scalar(self, name, value, step="frame"):
        '''
        Log an arbitrary scalar.
        Args:
            name (str): The tag to associate with the scalar
            value (number): The value of the scalar at the current step
            step (str, optional): Which step to use (e.g., "frame" or "episode")
        '''
        super().add_scalar(self.env_name + "/" + name, value, self._get_step(step))

    def add_summary(self, name, mean, std, step="frame"):
        self.add_evaluation(name + "/mean", mean, step)
        self.add_evaluation(name + "/std", std, step)

        with open(os.path.join(self.log_dir, self.env_name, name + ".csv"), "a") as csvfile:
            csv.writer(csvfile).writerow([self._get_step(step), mean, std])

    def _get_step(self, _type):
        if _type == "frame":
            return self._experiment.frame
        if _type == "episode":
            return self._experiment.episode
        return _type

def get_commit_hash():
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False
    )
    return result.stdout.decode("utf-8").rstrip()


COMMIT_HASH = get_commit_hash()
