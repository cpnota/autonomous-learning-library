from all.experiments import Experiment
from all.environments import AtariEnvironment
from all.presets.reinforce import reinforce_atari

def run():
    env = AtariEnvironment('Breakout')
    experiment = Experiment(env, episodes=100000, trials=1)
    experiment.run(
        reinforce_atari(),
        render=True
    )


if __name__ == '__main__':
    run()
