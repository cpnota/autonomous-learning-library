import csv
import os
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime
from tensorboardX import SummaryWriter


class Writer(ABC):
    log_dir = "runs"

    @abstractmethod
    def add_loss(self, name, value, step="frame"):
        '''
        Log the given loss metric at the current step.

        Args:
            name (str): The tag to associate with the loss
            value (number): The value of the loss at the current step
            step (str, optional): Which step to use (e.g., "frame" or "episode")
        '''

    @abstractmethod
    def add_evaluation(self, name, value, step="frame"):
        '''
        Log the evaluation metric.

        Args:
            name (str): The tag to associate with the loss
            value (number): The evaluation metric at the current step
            step (str, optional): Which step to use (e.g., "frame" or "episode")
        '''

    @abstractmethod
    def add_scalar(self, name, value, step="frame"):
        '''
        Log an arbitrary scalar.

        Args:
            name (str): The tag to associate with the scalar
            value (number): The value of the scalar at the current step
            step (str, optional): Which step to use (e.g., "frame" or "episode")
        '''

    @abstractmethod
    def add_schedule(self, name, value, step="frame"):
        '''
        Log the current value of a hyperparameter according to some schedule.

        Args:
            name (str): The tag to associate with the hyperparameter schedule
            value (number): The value of the hyperparameter at the current step
            step (str, optional): Which step to use (e.g., "frame" or "episode")
        '''

    @abstractmethod
    def add_summary(self, name, mean, std, step="frame"):
        '''
        Log a summary statistic.

        Args:
            name (str): The tag to associate with the summary statistic
            mean (float): The mean of the statistic at the current step
            std (float): The standard deviation of the statistic at the current step
            step (str, optional): Which step to use (e.g., "frame" or "episode")
        '''

    @abstractmethod
    def close(self):
        '''
        Close the writer and perform any necessary cleanup.
        '''


class DummyWriter(Writer):
    '''A default Writer object that performs no logging and has no side effects.'''

    def add_loss(self, name, value, step="frame"):
        pass

    def add_evaluation(self, name, value, step="frame"):
        pass

    def add_scalar(self, name, value, step="frame"):
        pass

    def add_schedule(self, name, value, step="frame"):
        pass

    def add_summary(self, name, mean, std, step="frame"):
        pass

    def close(self):
        pass
