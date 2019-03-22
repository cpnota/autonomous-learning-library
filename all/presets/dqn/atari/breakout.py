from all.experiments import Experiment
from all.environments import AtariEnvironment
from all.presets.dqn import dqn

def run():
    env = AtariEnvironment("Breakout")
    experiment = Experiment(env, frames=200e6, trials=1)
    experiment.run(
        dqn(),
        render=True
    )


if __name__ == '__main__':
    run()
