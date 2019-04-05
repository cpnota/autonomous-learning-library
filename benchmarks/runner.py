import subprocess
from all.experiments import Experiment

def run(agent_name, agent, env, episodes=None, frames=None):
    result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], stdout=subprocess.PIPE)
    rev = result.stdout.decode('utf-8')
    experiment = Experiment(
        env,
        frames=frames,
        episodes=episodes
    )
    experiment.run(agent, label=agent_name + " " + rev)
