from all.experiments import Experiment
from all.environments import AtariEnvironment
from all.presets.rainbow import rainbow

def run():
    env = AtariEnvironment("Breakout")
    experiment = Experiment(env, episodes=10000, trials=1)
    experiment.run(
        rainbow(),
        plot_every=10,
        print_every=1,
        render=True
    )
    experiment.plot(filename="rainbow-breakout.png", frequency=100)
    experiment.save("rainbow-breakout")


if __name__ == '__main__':
    run()
