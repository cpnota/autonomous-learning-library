from all.experiments import Experiment
from all.presets.reinforce import reinforce_cc

def run():
    experiment = Experiment('CartPole-v0', episodes=2000, trials=1)
    experiment.run(
        reinforce_cc(),
        plot_every=50,
        print_every=1,
        render=True
    )
    experiment.plot(filename="reinforce_cartpole")

if __name__ == '__main__':
    run()
