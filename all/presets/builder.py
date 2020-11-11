from abc import ABC, abstractmethod


class PresetBuilder(ABC):
    default_hyperparameters = None

    def __init__(self, hyperparameters={}, env=None, device='cuda'):
        if self.default_hyperparameters is None:
            raise AttributeError('default_hyperparameters must be defined')
        self._device = device
        self._env = env
        self._hyperparameters = {**self.default_hyperparameters, **hyperparameters}

    def hyperparameters(self, **hyperparameters):
        self._validate_hyperparameters(hyperparameters)
        return self.__class__(
            device=self._device,
            hyperparameters={**self.default_hyperparameters, **hyperparameters},
            env=self._env
        )

    def env(self, env):
        return self.__class__(
            device=self._device,
            hyperparameters=self._hyperparameters,
            env=env
        )

    def device(self, device):
        return self.__class__(device=device, hyperparameters=self._hyperparameters, env=self._env)

    def _validate_hyperparameters(self, hyperparameters):
        for key in hyperparameters.keys():
            if key not in self.default_hyperparameters:
                raise KeyError("Invalid hyperparameter: {}".format(key))

    @abstractmethod
    def build(self):
        pass
