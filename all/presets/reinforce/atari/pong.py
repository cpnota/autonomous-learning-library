from all.experiments import Experiment
from all.environments import make_atari
from all.presets.reinforce import reinforce_atari

def run():
    env = make_atari('PongNoFrameskip-v4')
    experiment = Experiment(env, episodes=20000, trials=1)
    experiment.run(
        reinforce_atari(),
        plot_every=50,
        print_every=1,
        render=True
    )
    experiment.plot(filename="reinforce-pong.png")
    experiment.save("reinforce_pong")


if __name__ == '__main__':
    run()
