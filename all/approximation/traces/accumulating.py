import numpy as np


class AccumulatingTraces:
    def __init__(self, approximation, env, decay_rate):
        self.env = env
        self.approximation = approximation
        self.decay_rate = decay_rate
        self.traces = np.zeros(approximation.parameters.shape)

    def call(self, *args):
        return self.approximation.call(*args)

    def update(self, error, *args):
        gradient = self.approximation.gradient(*args)
        self.traces += gradient
        self.approximation.apply(error * self.traces)
        self.traces *= self.decay_rate if not self.env.done else 0

    def gradient(self, *args):
        return self.approximation.gradient(*args)

    def update_parameters(self, gradient):
        # TODO
        pass

    def get_parameters(self):
        return self.approximation.parameters

    def set_parameters(self, parameters):
        self.approximation.parameters = parameters
