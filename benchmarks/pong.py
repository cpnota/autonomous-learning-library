import subprocess
import sys
from all.environments import AtariEnvironment
from all.experiments import Experiment
from all.presets import atari

def run():
    try:
        agent_name = sys.argv[1]
    except IndexError:
        print('Usage: python benchmarks/pong.py [agent_name]')
        exit(1)

    try:
        agent = getattr(atari, agent_name)
    except NameError:
        print('Unknown agent: ' + agent_name)
        exit(1)

    # compute hash so that results can be exactly reconstructed
    # by reverting to previous hash
    result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], stdout=subprocess.PIPE)
    rev = result.stdout.decode('utf-8')

    experiment = Experiment(
        AtariEnvironment('Pong'),
        frames=20e6
    )
    experiment.run(agent(), label=agent_name + "_" + rev)

if __name__ == '__main__':
    run()
