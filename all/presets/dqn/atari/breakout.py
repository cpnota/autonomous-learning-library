from all.experiments import Experiment
from all.environments import AtariEnvironment
from all.presets.dqn import dqn

def run():
    env = AtariEnvironment('Breakout')
    experiment = Experiment(env, episodes=100000, trials=1)
    experiment.run(
        dqn(),
        plot_every=100,
        print_every=1,
        render=True
    )
    experiment.plot(filename="dqn-breakout.png")
    experiment.save("dqn_breakout")


if __name__ == '__main__':
    run()
