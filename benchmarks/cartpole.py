import subprocess
import sys
from all.environments import GymEnvironment
from all.experiments import Experiment
from all.presets import classic_control

def run():
    try:
        agent_name = sys.argv[1]
    except IndexError:
        print('Usage: python benchmarks/cartpole.py [agent_name]')
        exit(1)

    try:
        agent = getattr(classic_control, agent_name)
    except NameError:
        print('Unknown agent: ' + agent_name)
        exit(1)


    result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], stdout=subprocess.PIPE)
    rev = result.stdout.decode('utf-8')

    experiment = Experiment(
        GymEnvironment('CartPole-v0'),
        episodes=1000
    )
    experiment.run(agent(), label=agent_name + " " + rev)

if __name__ == '__main__':
    run()
