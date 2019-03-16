from .gym import GymEnvironment

class AtariEnvironment(GymEnvironment):
    def __init__(self, env):
        self._name = env
        super().__init__(env + 'NoFrameskip-v4')

    def name(self):
        return self._name
