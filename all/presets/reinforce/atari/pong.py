from all.experiments import Experiment
# from all.environments import AtariEnvironment
from all.environments.pong import PongEnvironment
from all.presets.reinforce import reinforce_atari

def run():
    # env = AtariEnvironment('Pong')
    env = PongEnvironment()
    experiment = Experiment(env, episodes=10000, trials=1)
    experiment.run(
        reinforce_atari(
            lr_pi=1e-5,
            lr_v=1e-5
        ),
        plot_every=100,
        print_every=1,
        render=True
    )
    experiment.plot(filename="reinforce-pong.png", frequency=100)
    experiment.save("reinforce_pong")


if __name__ == '__main__':
    run()
