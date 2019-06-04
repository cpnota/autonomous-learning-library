import torch
from all.experiments import SlurmExperiment
from all.presets.atari import a2c
from all.environments import AtariEnvironment

device = torch.device('cuda')
envs = [AtariEnvironment(env, device) for env in ['Pong', 'Breakout', 'SpaceInvaders']]
SlurmExperiment(a2c, envs, 50000, hyperparameters={'device': device})
