from abc import ABC, abstractmethod


class Logger(ABC):
    log_dir = "runs"

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
    def add_loss(self, name, value, step="frame"):
        '''
        Log the given loss metric at the current step.

        Args:
            name (str): The tag to associate with the loss
            value (number): The value of the loss at the current step
            step (str, optional): Which step to use (e.g., "frame" or "episode")
        '''

    @abstractmethod
    def add_eval(self, name, value, step="frame"):
        '''
        Log the given evaluation metric at the current step.

        Args:
            name (str): The tag to associate with the loss
            value (number): The evaluation metric at the current step
            step (str, optional): Which step to use (e.g., "frame" or "episode")
        '''

    @abstractmethod
    def add_info(self, name, value, step="frame"):
        '''
        Log the given informational metric at the current step.

        Args:
            name (str): The tag to associate with the loss
            value (number): The evaluation metric at the current step
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
    def close(self):
        '''
        Close the logger and perform any necessary cleanup.
        '''
