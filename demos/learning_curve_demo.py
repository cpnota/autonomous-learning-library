from all.presets.tabular import sarsa
from all.presets.tabular import actor_critic
from all.experiments import Experiment

def run():
    experiment = Experiment('FrozenLake-v0')
    print('testing sarsa...')
    experiment.run(sarsa)
    print('testing actor_critic...')
    experiment.run(actor_critic)
    print('generating default plot...')
    experiment.plot()

if __name__ == '__main__':
    run()
