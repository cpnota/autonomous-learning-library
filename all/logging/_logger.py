from abc import ABC, abstractmethod


class Logger(ABC):
    log_dir = "runs"

    @abstractmethod
    def add_summary(self, name, mean, std, step="frame"):
        """
        Log a summary statistic.

        Args:
            name (str): The tag to associate with the summary statistic
            mean (float): The mean of the statistic at the current step
            std (float): The standard deviation of the statistic at the current step
            step (str, optional): Which step to use (e.g., "frame" or "episode")
        """

    @abstractmethod
    def add_loss(self, name, value, step="frame"):
        """
        Log the given loss metric at the current step.

        Args:
            name (str): The tag to associate with the loss
            value (number): The value of the loss at the current step
            step (str, optional): Which step to use (e.g., "frame" or "episode")
        """

    @abstractmethod
    def add_eval(self, name, value, step="frame"):
        """
        Log the given evaluation metric at the current step.

        Args:
            name (str): The tag to associate with the loss
            value (number): The evaluation metric at the current step
            step (str, optional): Which step to use (e.g., "frame" or "episode")
        """

    @abstractmethod
    def add_info(self, name, value, step="frame"):
        """
        Log the given informational metric at the current step.

        Args:
            name (str): The tag to associate with the loss
            value (number): The evaluation metric at the current step
            step (str, optional): Which step to use (e.g., "frame" or "episode")
        """

    @abstractmethod
    def add_schedule(self, name, value, step="frame"):
        """
        Log the current value of a hyperparameter according to some schedule.

        Args:
            name (str): The tag to associate with the hyperparameter schedule
            value (number): The value of the hyperparameter at the current step
            step (str, optional): Which step to use (e.g., "frame" or "episode")
        """

    @abstractmethod
    def add_hparams(self, hparam_dict, metric_dict, step="frame"):
        """
        Logs metrics for a given set of hyperparameters.
        Usually this should be called once at the end of a run in order to
        log the final results for hyperparameters, though it can be called
        multiple times throughout training. However, it should be called infrequently.

        Args:
            hparam_dict (dict): A dictionary of hyperparameters.
                Only parameters of type (int, float, str, bool, torch.Tensor)
                will be logged.
            metric_dict (dict): A dictionary of metrics to record.
            step (str, optional): Which step to use (e.g., "frame" or "episode")
        """
        pass

    @abstractmethod
    def close(self):
        """
        Close the logger and perform any necessary cleanup.
        """
