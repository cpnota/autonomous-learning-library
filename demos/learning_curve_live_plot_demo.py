from all.presets.tabular import sarsa
from all.experiments import Experiment

def run():
    learning_curve = Experiment('FrozenLake-v0', trials=1000)
    learning_curve.run(sarsa, plot_every=1000, print_every=1000)
    learning_curve.plot()

if __name__ == '__main__':
    run()
