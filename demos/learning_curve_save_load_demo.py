import os
from all.presets.tabular import sarsa
from all.presets.tabular import actor_critic
from all.experiments import Experiment, learning_curve

def run():
    # create some learning curve data
    experiment = Experiment('FrozenLake-v0')
    experiment.run(sarsa)
    experiment.run(actor_critic)

    # save the data to file
    experiment.save('lc.json')

    # load the data from file in new learning curve object
    results = Experiment.load('lc.json')
    learning_curve(results)

    # cleanup
    os.remove('lc.json')

if __name__ == '__main__':
    run()
