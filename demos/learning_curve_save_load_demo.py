import os
from all.presets.tabular import sarsa
from all.presets.tabular import actor_critic
from all.experiments import LearningCurve

def run():
    # create some learning curve data
    lc_original = LearningCurve('FrozenLake-v0')
    lc_original.run(sarsa)
    lc_original.run(actor_critic)

    # save the data to file
    lc_original.save('lc.json')

    # load the data from file in new learning curve object
    lc_loaded = LearningCurve('FrozenLake-v0')
    lc_loaded.load('lc.json')
    lc_loaded.plot()

    # cleanup
    os.remove('lc.json')

if __name__ == '__main__':
    run()
