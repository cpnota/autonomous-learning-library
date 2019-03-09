from all.experiments import Experiment
from all.presets.actor_critic import ac_cc

def run():
    experiment = Experiment('CartPole-v0', episodes=1000, trials=1)
    experiment.run(
        ac_cc(),
        plot_every=50,
        print_every=1,
        render=True
    )
    experiment.plot(filename="ac_cartpole", frequency=50)

if __name__ == '__main__':
    run()
