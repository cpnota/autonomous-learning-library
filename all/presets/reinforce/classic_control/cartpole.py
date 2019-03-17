from all.experiments import Experiment
from all.presets.reinforce import reinforce_cc

def run():
    experiment = Experiment('CartPole-v0', episodes=1000, trials=1)
    experiment.run(
        reinforce_cc(),
        render=True
    )

if __name__ == '__main__':
    run()
