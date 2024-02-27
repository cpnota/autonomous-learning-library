from ._logger import Logger


class DummyLogger(Logger):
    """A default Logger object that performs no logging and has no side effects."""

    def add_eval(self, name, value, step="frame"):
        pass

    def add_info(self, name, value, step="frame"):
        pass

    def add_loss(self, name, value, step="frame"):
        pass

    def add_schedule(self, name, value, step="frame"):
        pass

    def add_summary(self, name, values, step="frame"):
        pass

    def add_hparams(self, hparam_dict, metric_dict, step="frame"):
        pass

    def close(self):
        pass
