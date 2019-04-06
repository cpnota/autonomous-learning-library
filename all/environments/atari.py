from .gym import GymEnvironment

class AtariEnvironment(GymEnvironment):
    def __init__(self, env, *args, **kwargs):
        self._name = env
        super().__init__(env + 'NoFrameskip-v4', *args, **kwargs)

    @property
    def name(self):
        return self._name
