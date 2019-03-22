from all.experiments import Experiment
from all.presets.sarsa import sarsa_cc

def run():
    experiment = Experiment('CartPole-v0', episodes=1000, trials=1)
    experiment.run(
        sarsa_cc(),
        render=True
    )

if __name__ == '__main__':
    run()
