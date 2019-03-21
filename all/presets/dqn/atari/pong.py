from all.experiments import Experiment
from all.environments import AtariEnvironment
from all.presets.dqn import dqn

def run():
    # env = PongEnvironment()
    env = AtariEnvironment("Pong")
    experiment = Experiment(env, frames=50e6, trials=1)
    experiment.run(
        dqn(),
        render=True
    )


if __name__ == '__main__':
    run()
