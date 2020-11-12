from abc import ABC, abstractmethod

def preset_builder(default_name, default_hyperparameters, constructor):
    class PresetBuilder():
        def __init__(self, name=default_name, hyperparameters=default_hyperparameters, env=None, device='cuda'):
            if default_hyperparameters is None:
                raise AttributeError('default_hyperparameters must be defined')
            self._validate_hyperparameters(hyperparameters)
            self._name = name
            self._device = device
            self._env = env
            self._hyperparameters = {**default_hyperparameters, **hyperparameters}

        def name(self, name):
            return self.__class__(
                name=_name,
                device=self._device,
                hyperparameters=self._hyperparameters,
                env=self._env
            )

        def hyperparameters(self, **hyperparameters):
            return self.__class__(
                name=self._name,
                device=self._device,
                hyperparameters=hyperparameters,
                env=self._env
            )

        def env(self, env):
            return self.__class__(
                name=self._name,
                device=self._device,
                hyperparameters=self._hyperparameters,
                env=env
            )

        def device(self, device):
            return self.__class__(name=self._name, device=device, hyperparameters=self._hyperparameters, env=self._env)

        def _validate_hyperparameters(self, hyperparameters):
            for key in hyperparameters.keys():
                if key not in default_hyperparameters:
                    raise KeyError("Invalid hyperparameter: {}".format(key))

        def build(self):
            return constructor(self._hyperparameters, self._env, self._device)

    return PresetBuilder
