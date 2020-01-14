'''
Quick demo of a2c running on slurm, a distributed cluster.
Note that it only runs for 1 million frames.
For real experiments, you will surely need a modified version of this script.
'''
from gym import envs
from all.experiments import SlurmExperiment
from all.presets.atari import a2c
from all.environments import AtariEnvironment

device = 'cuda'
envs = [AtariEnvironment(env, device) for env in ['Pong', 'Breakout', 'SpaceInvaders']]
SlurmExperiment(a2c(device=device), envs, 1e6, sbatch_args={
    'partition': '1080ti-short'
})
