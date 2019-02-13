from all.experiments import Experiment
from all.environments import AtariEnvironment
from all.presets.reinforce import reinforce_atari

def run():
    env = AtariEnvironment('Breakout')
    experiment = Experiment(env, episodes=100000, trials=1)
    experiment.run(
        reinforce_atari(),
        plot_every=100,
        print_every=1,
        render=True
    )
    experiment.plot(filename="reinforce-breakout.png")
    experiment.save("reinforce_breakout")


if __name__ == '__main__':
    run()
