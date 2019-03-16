from all.experiments import Experiment
from all.environments import AtariEnvironment
from all.presets.dqn import dqn

def run():
    env = AtariEnvironment("Breakout")
    experiment = Experiment(env, episodes=40000, trials=1)
    experiment.run(
        dqn(),
        print_every=1,
        render=True
    )


if __name__ == '__main__':
    run()
