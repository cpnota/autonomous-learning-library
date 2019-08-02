'''
Run the rull atari suite on swarm with a2c.
You should modify this script to suit your needs.
'''
from gym import envs
from all.experiments import SlurmExperiment
from all.presets.atari import a2c
from all.environments import GymEnvironment

# Use the first available GPU
device = 'cuda'

# Get all atari envs.
# Note that this actually returns some games that aren't compatible.
# Those slurm tasks will simply fail.
envs = [
    GymEnvironment(env, device) for env in envs.registry.env_specs.keys()
    if 'NoFrameskip-v4' in env and not '-ram' in env
]

SlurmExperiment(a2c(device=device), envs, 1e9, sbatch_args={
    'partition': '1080ti-long' # long queue: run for a week
})
