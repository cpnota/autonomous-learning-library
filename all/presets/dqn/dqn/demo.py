from all.experiments import Experiment
from all.environments import make_atari
from all.presets.dqn import dqn

def run():
    env = make_atari('BreakoutDeterministic-v4')
    experiment = Experiment(env, episodes=1000000, trials=1)
    experiment.run(
        dqn,
        plot_every=1,
        print_every=1,
        render=True
    )
    experiment.plot(filename="dqn-breakout")


if __name__ == '__main__':
    run()
