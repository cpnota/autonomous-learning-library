from all.experiments import Experiment
from all.environments import AtariEnvironment
from all.presets.dqn import dqn

def run():
    env = AtariEnvironment('Pong')
    experiment = Experiment(env, episodes=2000, trials=1)
    experiment.run(
        dqn(),
        plot_every=50,
        print_every=1,
        render=True
    )
    experiment.plot(filename="dqn-pong.png")
    experiment.save("dqn_pong")


if __name__ == '__main__':
    run()
