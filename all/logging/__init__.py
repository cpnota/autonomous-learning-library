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
        pass

    @abstractmethod
    def add_evaluation(self, name, value, step="frame"):
        pass

    @abstractmethod
    def add_scalar(self, name, value, step="frame"):
        pass

    @abstractmethod
    def add_schedule(self, name, value, step="frame"):
        pass

    @abstractmethod
    def add_summary(self, name, mean, std, step="frame"):
        pass


class DummyWriter(Writer):
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

