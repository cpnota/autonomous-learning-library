from all.presets.tabular import sarsa
from all.presets.tabular import actor_critic
from all.experiments import Experiment

def run():
    experiment = Experiment('FrozenLake-v0')
    experiment.run(sarsa)
    experiment.run(actor_critic)
    # save default plot to file
    experiment.plot(filename="demo_plot.png")

if __name__ == '__main__':
    run()
