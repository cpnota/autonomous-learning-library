from all.experiments import Experiment
from all.presets.sarsa import sarsa_cc

def run():
    experiment = Experiment('CartPole-v0', episodes=2000, trials=1)
    experiment.run(
        sarsa_cc(),
        plot_every=50,
        print_every=1,
        render=True
    )
    experiment.plot(filename="sarsa_cartpole")

if __name__ == '__main__':
    run()
