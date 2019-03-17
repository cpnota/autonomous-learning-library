from all.experiments import Experiment
from all.environments import AtariEnvironment
from all.presets.rainbow import rainbow

def run():
    env = AtariEnvironment("Breakout")
    experiment = Experiment(env, episodes=100000, trials=1)
    experiment.run(
        rainbow(),
        render=True
    )


if __name__ == '__main__':
    run()
