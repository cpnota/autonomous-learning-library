from all.experiments import SlurmExperiment
from all.presets.atari import a2c
from all.environments import AtariEnvironment

envs = [AtariEnvironment(env) for env in ['Pong', 'Breakout', 'SpaceInvaders']]

SlurmExperiment(a2c, envs)
