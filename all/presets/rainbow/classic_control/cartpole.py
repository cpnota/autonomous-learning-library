from all.experiments import Experiment
from all.presets.rainbow import rainbow_cc

def run():
    experiment = Experiment('CartPole-v0', episodes=1000, trials=1)
    experiment.run(
        rainbow_cc(),
        print_every=1,
        render=True
    )

if __name__ == '__main__':
    run()
