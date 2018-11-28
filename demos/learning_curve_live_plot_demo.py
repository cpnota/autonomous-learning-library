from all.presets.tabular import sarsa
from all.experiments import Experiment

def run():
    experiment = Experiment('FrozenLake-v0', trials=1000)
    # generate default plot every 1000 episodes, and print progress
    experiment.run(sarsa, plot_every=1000, print_every=1000)

if __name__ == '__main__':
    run()
