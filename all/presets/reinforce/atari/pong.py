from all.experiments import Experiment
from all.environments import AtariEnvironment
from all.presets.reinforce import reinforce_atari

def run():
    env = AtariEnvironment('Pong')
    experiment = Experiment(env, episodes=10000, trials=1)
    experiment.run(
        reinforce_atari(
            lr_pi=1e-5,
            lr_v=1e-5
        ),
        render=True
    )

if __name__ == '__main__':
    run()
