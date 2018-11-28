from all.presets.tabular import sarsa
from all.presets.tabular import actor_critic
from all.experiments import Experiment, return_distribution


def run():
    experiment = Experiment('FrozenLake-v0', trials=50)
    print('testing sarsa...')
    experiment.run(sarsa)
    print('testing actor_critic...')
    experiment.run(actor_critic)
    print('generating return distribution plot...')
    experiment.plot(return_distribution)

if __name__ == '__main__':
    run()
