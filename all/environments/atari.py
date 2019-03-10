from .gym import GymEnvironment

class AtariEnvironment(GymEnvironment):
    def __init__(self, env):
        super().__init__(env + 'NoFrameskip-v4')
