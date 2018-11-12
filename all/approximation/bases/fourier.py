import numpy as np
from all.approximation.bases.basis import Basis

class FourierBasis(Basis):
    def __init__(self, space, max_frequency):
        inputs = space.shape[0]
        scale = space.high - space.low
        self.offset = -space.low

        # 0th order (constant)
        self.weights = np.zeros(inputs)

        # first order correlations
        for i in range(inputs):
            for frequency in range(max_frequency):
                row = np.zeros(inputs)
                row[i] = frequency + 1
                self.weights = np.vstack([self.weights, row])

        # second order correlations
        for i in range(inputs):
            for j in range(i, inputs):
                if i == j:
                    continue
                for i_freq in range(max_frequency):
                    for j_freq in range(max_frequency):
                        row = np.zeros(inputs)
                        row[i] = i_freq + 1
                        row[j] = j_freq + 1
                        self.weights = np.vstack([self.weights, row])

        self.weights *= np.pi
        self.weights /= scale
        self._num_features = self.weights.shape[0]

    def features(self, args):
        return np.cos(self.weights.dot(args + self.offset))

    @property
    def num_features(self):
        return self._num_features
