import subprocess
import sys
import torch
from all.experiments import Experiment

def run(presets, env, episodes=None, frames=None):
    try:
        agent_name = sys.argv[1]
    except IndexError:
        print('Usage: python benchmarks/[env].py [agent]')
        exit(1)

    try:
        agent = getattr(presets, agent_name)
    except NameError:
        print('Unknown agent: ' + agent_name)
        exit(1)


    result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], stdout=subprocess.PIPE)
    rev = result.stdout.decode('utf-8')

    device = torch.device('cuda')

    env._device = device

    experiment = Experiment(
        env,
        frames=frames,
        episodes=episodes
    )
    experiment.run(agent(device=device), label=agent_name + " " + rev)
