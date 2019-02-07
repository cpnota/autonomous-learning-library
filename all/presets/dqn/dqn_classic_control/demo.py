from all.experiments import Experiment
from all.presets.dqn import dqn_cc

def run():
    experiment = Experiment('CartPole-v0', episodes=2000, trials=1)
    experiment.run(
        dqn_cc,
        plot_every=1,
        print_every=1,
        render=True
    )
    experiment.plot(filename="dqn_cartpole")

if __name__ == '__main__':
    run()
