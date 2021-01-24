from abc import ABC, abstractmethod


class PresetBuilder():
    def __init__(
        self,
        default_name,
        default_hyperparameters,
        constructor,
        device="cuda",
        env=None,
        hyperparameters=None,
        name=None,
    ):
        self.default_name = default_name
        self.default_hyperparameters = default_hyperparameters
        self.constructor = constructor
        self._device = device
        self._env = env
        self._hyperparameters = self._merge_hyperparameters(default_hyperparameters, hyperparameters)
        self._name = name or default_name

    def device(self, device):
        return self._preset_builder(device=device)

    def env(self, env):
        return self._preset_builder(env=env)

    def hyperparameters(self, **hyperparameters):
        return self._preset_builder(hyperparameters=self._merge_hyperparameters(self._hyperparameters, hyperparameters))

    def name(self, name):
        return self._preset_builder(name=name)

    def build(self):
        if not self._env:
            raise Exception('Env is required')

        return self.constructor(
            self._env,
            device=self._device,
            name=self._name,
            **self._hyperparameters
        )

    def _merge_hyperparameters(self, h1, h2):
        if h2 is None:
            return h1
        for key in h2.keys():
            if key not in h1:
                raise KeyError("Invalid hyperparameter: {}".format(key))
        return {**h1, **h2}

    def _preset_builder(self, **kwargs):
        old_kwargs = {
            'device': self._device,
            'env': self._env,
            'hyperparameters': self._hyperparameters,
            'name': self._name,
        }
        return PresetBuilder(self.default_name, self.default_hyperparameters, self.constructor, **{**old_kwargs, **kwargs})


class ParallelPresetBuilder(PresetBuilder):
    def __init__(
        self,
        default_name,
        default_hyperparameters,
        constructor,
        device="cuda",
        env=None,
        hyperparameters=None,
        name=None,
    ):
        if 'n_envs' not in default_hyperparameters:
            raise Exception('ParallelPreset hyperparameters must include n_envs')
        super().__init__(
            default_name,
            default_hyperparameters,
            constructor,
            device=device,
            env=env,
            hyperparameters=hyperparameters,
            name=name
        )

    def build(self):
        return super().build()
