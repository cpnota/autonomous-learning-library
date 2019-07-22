'''Create slurm tasks to run benchmark suite'''
import argparse
from all.environments import AtariEnvironment, GymEnvironment
from all.experiments import SlurmExperiment
from all.presets import atari, classic_control, continuous

# run on gpu
device = 'cuda'

# create slurm tasks for running classic control agents
for agent_name in classic_control.__all__:
    print('CartPole-v0,', agent_name)
    agent = getattr(classic_control, agent_name)
    envs = [GymEnvironment('CartPole-v0', device=device)]
    SlurmExperiment(agent, envs, 100000, hyperparameters={'device': device}, sbatch_args={
        'partition': '1080ti-short'
    })

# create slurm tasks for running atari agents
for agent_name in atari.__all__:
    print('Breakout', agent_name)
    agent = getattr(atari, agent_name)
    envs = [AtariEnvironment('Breakout', device=device)]
    SlurmExperiment(agent, envs, 2e7, hyperparameters={'device': device}, sbatch_args={
        'partition': '1080ti-long'
    })

# create slurm tasks for running atari agents
for agent_name in continuous.__all__:
    print('Lander', agent_name)
    agent = getattr(continuous, agent_name)
    envs = [GymEnvironment('LunarLanderContinuous-v2', device=device)]
    SlurmExperiment(agent, envs, 500000, hyperparameters={'device': device}, sbatch_args={
        'partition': '1080ti-short'
    })
