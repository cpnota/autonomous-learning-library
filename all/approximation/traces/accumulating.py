import numpy as np
from all.approximation import Approximation


class AccumulatingTraces(Approximation):
    def __init__(self, approximation, env, decay_rate):
        self.env = env
        self.approximation = approximation
        self.decay_rate = decay_rate
        self.traces = np.zeros(approximation.parameters.shape)

    def __call__(self, *args):
        return self.approximation(*args)

    def update(self, error, *args):
        gradient = self.approximation.gradient(*args)
        self.traces += gradient
        self.approximation.apply(error * self.traces)
        self.traces *= self.decay_rate if not self.env.done else 0

    def gradient(self, *args):
        return self.approximation.gradient(*args)

    def apply(self, gradient):
        return self.approximation.apply(gradient)

    @property
    def parameters(self):
        return self.approximation.parameters

    @parameters.setter
    def parameters(self, parameters):
        self.approximation.parameters = parameters
