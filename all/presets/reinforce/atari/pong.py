from all.experiments import Experiment
# from all.environments import AtariEnvironment
from all.environments.pong import PongEnvironment
from all.presets.reinforce import reinforce_atari

def run():
    # env = AtariEnvironment('Pong')
    env = PongEnvironment()
    experiment = Experiment(env, episodes=20000, trials=1)
    experiment.run(
        reinforce_atari(),
        plot_every=100,
        print_every=1,
        render=True
    )
    experiment.plot(filename="reinforce-pong.png")
    experiment.save("reinforce_pong")


if __name__ == '__main__':
    run()
