import torch
from gym import envs
from all.experiments import SlurmExperiment
from all.presets.atari import a2c
from all.environments import GymEnvironment

device = torch.device('cuda')
envs = [GymEnvironment(env, device) for env in envs.registry.env_specs.keys() if 'NoFrameskip-v4' in env and not '-ram' in env]
# envs = [AtariEnvironment(env, device) for env in ['Pong', 'Breakout', 'SpaceInvaders']]
SlurmExperiment(a2c, envs, 50000, hyperparameters={'device': device})
